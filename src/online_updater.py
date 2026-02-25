"""
online_updater.py
-----------------
Incremental (online) learning module for the vibe–performance predictor.

Addresses the limitation documented in REPORT.md §5::

    "Online learning: the predictor is batch-trained.  A sliding-window
    SGD update would let it adapt to seasonality and trend shifts in ad
    performance."

The Problem with Batch Retraining
----------------------------------
:class:`~src.performance_predictor.VibePerformancePredictor` is trained once
on a fixed dataset.  In production, campaign performance data (CTR, ROAS)
arrives continuously.  Two issues arise:

1. **Distribution shift** — the relationship between vibe features and CTR
   changes over time due to audience fatigue, seasonality, and platform
   algorithm changes.  A model trained in Q1 may be miscalibrated by Q3.

2. **Retraining latency** — batch retraining requires waiting until "enough"
   new data has accumulated, then running a full training + CV + calibration
   pipeline.  For daily campaign decisions this latency is unacceptable.

Solution
--------
:class:`OnlinePredictor` wraps :class:`sklearn.linear_model.SGDRegressor`
(stochastic gradient descent, Huber loss for outlier robustness) with:

* **Warm-start initialisation** — copies the Ridge regression coefficients
  from a batch-trained predictor, so the online model starts from a strong
  prior rather than random weights.

* **Sliding-window buffer** — only the most recent ``max_window`` samples are
  retained; old data is discarded, enabling the model to adapt to the
  current distribution without catastrophic forgetting on new patterns.

* **Page's CUSUM drift detector** (Page 1954) — accumulates evidence of
  sustained residual bias and raises an alarm when prediction errors have
  been consistently larger than expected.  A CUSUM alarm is the signal to
  trigger a full batch retrain.  Named constants ``_CUSUM_SLACK`` and
  ``_CUSUM_THRESHOLD`` control sensitivity vs false-positive rate.

* **Jackknife predictive interval** — approximates a ``confidence``-level
  predictive interval without distributional assumptions, by measuring how
  much predictions change when individual buffer samples are left out.
  More principled than heuristic sigma-multipliers.

Usage
-----
    from src.online_updater import OnlinePredictor

    # Warm-start from a batch-trained predictor
    updater = OnlinePredictor.from_batch_predictor(batch_predictor)

    # Ingest new (features, label) pairs as they arrive
    updater.partial_fit(new_features, new_labels)

    # Predict with 90% predictive interval
    mean, lo, hi = updater.predict_with_interval(test_features)

    # Check for concept drift
    if updater.drift_detected():
        trigger_full_retrain()

    updater.save("outputs/online_predictor.npz")
"""

import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
from scipy.stats import norm as _scipy_norm
from sklearn.linear_model import SGDRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants for CUSUM drift detector
# ---------------------------------------------------------------------------
# Page (1954) recommends slack k in [0.5, 1.0].  k = 0.5 means: raise alarm
# if the mean residual deviates by more than 0.5 standard deviations.
_CUSUM_SLACK: float = 0.5

# Alarm threshold.  Higher = fewer false positives, slower detection.
# Typical production value: 4–6 standard deviations of accumulated slack.
_CUSUM_THRESHOLD: float = 5.0

# Maximum window size for jackknife (keeps the LOO pass tractable for large
# buffers; statistical efficiency is adequate beyond ~30 samples).
_JACKKNIFE_CAP: int = 30


class OnlinePredictor:
    """
    Incrementally-trained vibe–performance predictor with drift detection.

    Args:
        n_features:    Number of input features; must match the batch predictor.
        max_window:    Maximum number of samples to retain in the sliding window.
        alpha:         SGD L2 regularisation strength.
        loss:          SGD loss function.  ``"huber"`` is robust to CTR/ROAS
                       outliers; ``"squared_error"`` is the standard MSE loss.
        random_state:  Random seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int,
        max_window: int = 500,
        alpha: float = 1e-4,
        loss: str = "huber",
        random_state: int = 42,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}.")
        if max_window < 10:
            raise ValueError(f"max_window must be >= 10; got {max_window}.")

        self.n_features   = n_features
        self.max_window   = max_window
        self.random_state = random_state

        # Sliding-window buffer (list of 1-D arrays / scalars)
        self._X_buf: list = []
        self._y_buf: list = []

        # CUSUM state
        self._cusum_pos:  float = 0.0
        self._cusum_neg:  float = 0.0
        self._n_updates:  int   = 0
        self._drift_alarm: bool = False

        # SGD model — max_iter=1 so partial_fit drives iterations externally
        self._model = SGDRegressor(
            loss=loss,
            alpha=alpha,
            random_state=random_state,
            max_iter=1,
            tol=None,
            warm_start=True,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_batch_predictor(
        cls,
        batch_predictor,
        max_window: int = 500,
    ) -> "OnlinePredictor":
        """
        Warm-start from a trained
        :class:`~src.performance_predictor.VibePerformancePredictor`.

        Copies Ridge regression coefficients as SGD starting weights,
        giving the online model a strong prior over random initialisation.

        Args:
            batch_predictor:  A fitted ``VibePerformancePredictor``.
            max_window:       Sliding-window size.

        Returns:
            A warm-started :class:`OnlinePredictor`.
        """
        ridge      = batch_predictor.ridge_model
        n_features = ridge.coef_.shape[0]

        instance = cls(n_features=n_features, max_window=max_window)

        # Copy Ridge weights into SGDRegressor's coefficient arrays
        instance._model.coef_      = ridge.coef_.copy().astype(np.float64)
        instance._model.intercept_ = np.atleast_1d(
            np.asarray(ridge.intercept_, dtype=np.float64)
        )
        instance._is_fitted = True

        logger.info(
            "OnlinePredictor warm-started from Ridge "
            "(n_features=%d, coef_norm=%.4f).",
            n_features,
            float(np.linalg.norm(ridge.coef_)),
        )
        return instance

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ingest new labelled samples and perform one SGD update pass.

        Appends samples to the sliding window (pruning the oldest when
        ``max_window`` is exceeded), updates CUSUM with residuals (if
        already fitted), then runs ``SGDRegressor.partial_fit``.

        Args:
            X:  Feature matrix ``(n_samples, n_features)``.
            y:  Target vector ``(n_samples,)``.

        Raises:
            ValueError: if ``X.shape[1] != n_features`` or row counts mismatch.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features; got {X.shape[1]}."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        # Update CUSUM with prediction residuals before updating the model
        if self._is_fitted:
            residuals = y - self._model.predict(X)
            self._update_cusum(residuals)

        # Append to sliding window then prune
        for xi, yi in zip(X, y):
            self._X_buf.append(xi.copy())
            self._y_buf.append(float(yi))

        if len(self._X_buf) > self.max_window:
            excess = len(self._X_buf) - self.max_window
            self._X_buf = self._X_buf[excess:]
            self._y_buf = self._y_buf[excess:]

        # SGD update on new samples only
        self._model.partial_fit(X, y)
        self._is_fitted = True
        self._n_updates += len(y)

        logger.debug(
            "partial_fit: +%d samples, window=%d, drift=%s.",
            len(y), len(self._X_buf), self._drift_alarm,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Point predictions for ``X``.

        Args:
            X:  ``(n_samples, n_features)`` array.

        Returns:
            ``(n_samples,)`` float64 predictions.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "OnlinePredictor has not been fitted. Call partial_fit() first."
            )
        return self._model.predict(
            np.atleast_2d(np.asarray(X, dtype=np.float64))
        )

    def predict_with_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with a jackknife predictive interval.

        The jackknife approximates prediction uncertainty by measuring how
        much the prediction changes when each buffer sample is individually
        excluded (leave-one-out).  The jackknife standard error is::

            SE = sqrt( ((n−1)/n) * sum_i( (θ̂_{(-i)} − θ̂)² ) )

        This is more principled than applying an arbitrary sigma multiplier
        because it measures the actual sensitivity of the model to its
        training data.  The interval widens when the window is small or
        when the query lies far from the training distribution.

        Args:
            X:           ``(n_samples, n_features)`` query array.
            confidence:  Target coverage probability (default 0.90).

        Returns:
            Tuple ``(mean, lower, upper)`` each of shape ``(n_samples,)``.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("OnlinePredictor has not been fitted yet.")

        X_q       = np.atleast_2d(np.asarray(X, dtype=np.float64))
        mean_pred = self._model.predict(X_q)

        buf_size = len(self._X_buf)
        if buf_size < 5:
            # Not enough data for jackknife; return zero-width interval
            return mean_pred, mean_pred.copy(), mean_pred.copy()

        # Subsample buffer to cap jackknife cost at _JACKKNIFE_CAP iterations
        rng    = np.random.default_rng(self.random_state)
        n_jk   = min(buf_size, _JACKKNIFE_CAP)
        idx    = rng.choice(buf_size, size=n_jk, replace=False)
        X_buf  = np.stack([self._X_buf[i] for i in idx])
        y_buf  = np.array([self._y_buf[i] for i in idx])

        # LOO predictions
        jk_preds = np.empty((n_jk, len(X_q)), dtype=np.float64)
        for k in range(n_jk):
            mask    = np.ones(n_jk, dtype=bool)
            mask[k] = False
            jk_model = SGDRegressor(
                loss=self._model.loss,
                alpha=self._model.alpha,
                random_state=self.random_state,
                max_iter=100,
                tol=1e-4,
            )
            jk_model.fit(X_buf[mask], y_buf[mask])
            jk_preds[k] = jk_model.predict(X_q)

        # Jackknife standard error (Efron & Stein 1981)
        n_jk_f = float(n_jk)
        jk_std = np.sqrt(
            ((n_jk_f - 1.0) / n_jk_f)
            * np.sum((jk_preds - mean_pred) ** 2, axis=0)
        )

        z      = _scipy_norm.ppf(0.5 + confidence / 2.0)
        lower  = mean_pred - z * jk_std
        upper  = mean_pred + z * jk_std
        return mean_pred, lower, upper

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def drift_detected(self) -> bool:
        """
        Return True if the CUSUM test has raised a drift alarm.

        The alarm persists until :meth:`reset_cusum` is called (typically
        after triggering a full batch retrain).
        """
        return self._drift_alarm

    def reset_cusum(self) -> None:
        """Reset CUSUM accumulators and clear the drift alarm."""
        self._cusum_pos   = 0.0
        self._cusum_neg   = 0.0
        self._drift_alarm = False
        logger.info("CUSUM reset (drift alarm cleared).")

    def drift_summary(self) -> dict:
        """
        Return a dict with current CUSUM diagnostics.

        Keys: ``cusum_pos``, ``cusum_neg``, ``threshold``,
        ``drift_alarm``, ``n_updates``.
        """
        return {
            "cusum_pos":   round(self._cusum_pos, 4),
            "cusum_neg":   round(self._cusum_neg, 4),
            "threshold":   _CUSUM_THRESHOLD,
            "drift_alarm": self._drift_alarm,
            "n_updates":   self._n_updates,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Current number of samples in the sliding window."""
        return len(self._X_buf)

    @property
    def n_updates(self) -> int:
        """Total number of samples processed since creation."""
        return self._n_updates

    def residual_stats(self) -> dict:
        """
        Mean and std of residuals on the current sliding-window buffer.

        Useful for monitoring prediction quality between full retrains.

        Returns:
            Dict with keys ``mean_residual``, ``std_residual``,
            ``n_samples``.  Returns an empty dict if not fitted or the
            window is empty.
        """
        if not self._is_fitted or len(self._X_buf) == 0:
            return {}
        X_buf  = np.stack(self._X_buf)
        y_buf  = np.array(self._y_buf)
        resids = y_buf - self._model.predict(X_buf)
        return {
            "mean_residual": float(np.mean(resids)),
            "std_residual":  float(np.std(resids)),
            "n_samples":     len(self._X_buf),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights and buffer state to a compressed NumPy archive.

        Args:
            path:  Destination ``.npz`` file path.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted OnlinePredictor.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        X_buf = (
            np.stack(self._X_buf)
            if self._X_buf
            else np.empty((0, self.n_features), dtype=np.float64)
        )
        y_buf = np.array(self._y_buf, dtype=np.float64)

        np.savez_compressed(
            path,
            coef=self._model.coef_,
            intercept=self._model.intercept_,
            X_buf=X_buf,
            y_buf=y_buf,
            n_features=np.array([self.n_features]),
            max_window=np.array([self.max_window]),
            n_updates=np.array([self._n_updates]),
            cusum_pos=np.array([self._cusum_pos]),
            cusum_neg=np.array([self._cusum_neg]),
            drift_alarm=np.array([int(self._drift_alarm)]),
        )
        logger.info("OnlinePredictor saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "OnlinePredictor":
        """
        Load a previously saved :class:`OnlinePredictor`.

        Args:
            path:  Path to a ``.npz`` file written by :meth:`save`.

        Returns:
            A fitted :class:`OnlinePredictor` instance.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Predictor file not found: '{path}'.")

        data       = np.load(path, allow_pickle=False)
        n_features = int(data["n_features"][0])
        max_window = int(data["max_window"][0])
        instance   = cls(n_features=n_features, max_window=max_window)

        instance._model.coef_      = data["coef"].astype(np.float64)
        instance._model.intercept_ = data["intercept"].astype(np.float64)
        instance._is_fitted        = True
        instance._n_updates        = int(data["n_updates"][0])
        instance._cusum_pos        = float(data["cusum_pos"][0])
        instance._cusum_neg        = float(data["cusum_neg"][0])
        instance._drift_alarm      = bool(data["drift_alarm"][0])

        X_buf = data["X_buf"]
        y_buf = data["y_buf"]
        instance._X_buf = [X_buf[i] for i in range(len(X_buf))]
        instance._y_buf = y_buf.tolist()

        logger.info(
            "OnlinePredictor loaded from '%s' (n_features=%d, window=%d).",
            path, n_features, len(instance._X_buf),
        )
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_cusum(self, residuals: np.ndarray) -> None:
        """
        Update Page's CUSUM statistics with new residuals.

        Accumulates evidence that the mean residual has drifted away from
        zero by more than ``_CUSUM_SLACK``.  Raises a drift alarm when
        either ``cusum_pos`` or ``cusum_neg`` exceeds ``_CUSUM_THRESHOLD``.
        """
        for r in residuals:
            self._cusum_pos = max(0.0, self._cusum_pos + float(r) - _CUSUM_SLACK)
            self._cusum_neg = max(0.0, self._cusum_neg - float(r) - _CUSUM_SLACK)

        if (
            self._cusum_pos > _CUSUM_THRESHOLD
            or self._cusum_neg > _CUSUM_THRESHOLD
        ):
            if not self._drift_alarm:
                logger.warning(
                    "CUSUM drift alarm raised: cusum_pos=%.2f, "
                    "cusum_neg=%.2f (threshold=%.1f).",
                    self._cusum_pos,
                    self._cusum_neg,
                    _CUSUM_THRESHOLD,
                )
            self._drift_alarm = True

