"""
calibration.py
--------------
Post-hoc probability calibration for the CTR/ROAS predictor.

Problem
~~~~~~~
``VibePerformancePredictor`` outputs raw regression scores.  A score of 0.7
does *not* mean "this creative will land in the top 30%".  The mapping from
raw scores to empirical frequencies is typically compressed: the model is
over-confident near 0 and 1, and under-confident in the middle.

This module provides post-hoc calibration that corrects this distortion so
that reported probabilities have a proper frequentist interpretation:
a score of 0.7 from the calibrated model means that ~70% of creatives with
that score actually land above the median CTR.

Two calibration methods are available:

1. **Platt scaling** — fits a logistic sigmoid ``P(above_median) =
   σ(a·x + b)`` to the held-out predictions.  Fast and reliable for monotone
   miscalibration.  Requires ≥ 20 held-out samples.

2. **Isotonic regression** — non-parametric monotone regression.  More
   flexible but tends to overfit for N < 50; recommended for N > 100.

Evaluation
~~~~~~~~~~
The **Expected Calibration Error (ECE)** summarises the reliability diagram
as a single number::

    ECE = Σ_b (|B_b| / N) · |mean_pred(B_b) − freq_pos(B_b)|

ECE < 0.05 is generally considered well-calibrated for ad-scoring systems.
A perfectly calibrated model has ECE = 0.

Usage
-----
    from src.calibration import PredictorCalibrator

    # Fit on out-of-fold cross-validated predictions
    cal = PredictorCalibrator(method="platt")
    cal.fit(oof_predictions, oof_labels)

    # Transform raw scores
    calibrated = cal.transform(raw_scores)

    ece = cal.expected_calibration_error(calibrated, labels)
    cal.plot_reliability_diagram(calibrated, labels, raw_scores=raw_scores,
                                 output_path="outputs/calibration.png")
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Very weak L2 regularisation for Platt scaling — we want the logistic
# regression to follow the data as closely as possible (the calibration
# mapping is already constrained to be monotone by the sorted input structure).
_PLATT_C: float = 1e10


class PredictorCalibrator:
    """
    Post-hoc calibration for a regression-based ad performance predictor.

    Fits a monotone calibration mapping from raw predictor scores to
    ``[0, 1]`` probabilities so that score = 0.7 means roughly 70% of
    creatives with that score perform above median CTR/ROAS.

    **Critical design note**: always fit calibration on *held-out*
    (out-of-fold) predictions.  Fitting on in-sample predictions produces
    trivially over-confident calibration: the model has already memorised
    those points, so its scores are over-compressed toward the correct labels.
    Use the ``fold_predictions`` collected during ``cross_validate()`` for
    honest calibration.

    Args:
        method:  ``"platt"`` (default) — sigmoid scaling via logistic
                 regression; well-suited for N ≥ 20 held-out samples.
                 ``"isotonic"`` — non-parametric monotone regression;
                 recommended for N ≥ 100.
    """

    def __init__(self, method: str = "platt") -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(
                f"Unknown method '{method}'; use 'platt' or 'isotonic'."
            )
        self.method = method
        self._calibrator = None
        self._is_fitted: bool = False
        self._n_train: int = 0
        self._threshold: float = 0.5  # stored for consistent binarisation

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> "PredictorCalibrator":
        """
        Fit the calibration mapping from held-out predictions to labels.

        Labels are binarised at their median: a creative is "positive" if its
        label exceeds the median.  This maps the regression output to a binary
        classification probability, which is what Platt/isotonic calibration
        expects.

        Args:
            predictions:  ``(N,)`` raw model predictions from held-out fold.
            labels:       ``(N,)`` ground-truth performance values.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: if ``len(predictions) < 2``.
        """
        if len(predictions) < 2:
            raise ValueError("Need ≥ 2 samples to fit calibrator.")

        self._threshold = float(np.median(labels))
        binary_labels = (labels > self._threshold).astype(np.int32)

        n_pos = int(binary_labels.sum())
        n_neg = int(len(binary_labels) - n_pos)

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Calibrator: all labels are the same class (%d pos, %d neg). "
                "Calibration will be trivial.",
                n_pos, n_neg,
            )

        if self.method == "platt":
            self._calibrator = LogisticRegression(C=_PLATT_C, solver="lbfgs")
            self._calibrator.fit(predictions.reshape(-1, 1), binary_labels)
        else:  # isotonic
            if len(predictions) < 50:
                logger.warning(
                    "Isotonic calibration with N=%d (<50) samples may overfit. "
                    "Consider method='platt'.",
                    len(predictions),
                )
            self._calibrator = IsotonicRegression(
                out_of_bounds="clip", increasing=True
            )
            self._calibrator.fit(predictions, binary_labels)

        self._is_fitted = True
        self._n_train = len(predictions)
        logger.info(
            "PredictorCalibrator (%s) fitted: N=%d, %d pos / %d neg, "
            "median_threshold=%.4f.",
            self.method, len(predictions), n_pos, n_neg, self._threshold,
        )
        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Map raw predictor scores to calibrated probabilities.

        Args:
            predictions:  ``(M,)`` raw model output scores.

        Returns:
            Float32 ``(M,)`` calibrated probabilities in ``[0, 1]``.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
        """
        if not self._is_fitted or self._calibrator is None:
            raise RuntimeError("Call fit() before transform().")

        if self.method == "platt":
            proba = self._calibrator.predict_proba(
                predictions.reshape(-1, 1)
            )[:, 1]
        else:
            proba = self._calibrator.predict(predictions)

        return np.clip(proba, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # ECE metric
    # ------------------------------------------------------------------

    def expected_calibration_error(
        self,
        calibrated_scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute the Expected Calibration Error (ECE).

        Uses equal-width bins over ``[0, 1]``.  For small datasets
        (N < 100) the ECE estimate is noisy; in that case use fewer bins
        or prefer reliability diagram visual inspection.

        Args:
            calibrated_scores:  ``(N,)`` calibrated probabilities.
            labels:             ``(N,)`` ground-truth values.
            n_bins:             Number of calibration bins.

        Returns:
            Float ≥ 0.  ECE < 0.05 is well-calibrated for most applications.
        """
        threshold = float(np.median(labels))
        binary = (labels > threshold).astype(np.float32)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(
            np.digitize(calibrated_scores, bins) - 1, 0, n_bins - 1
        )

        ece = 0.0
        n = len(calibrated_scores)
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            n_b = int(mask.sum())
            mean_pred = float(calibrated_scores[mask].mean())
            mean_actual = float(binary[mask].mean())
            ece += (n_b / n) * abs(mean_pred - mean_actual)

        return float(ece)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_reliability_diagram(
        self,
        calibrated_scores: np.ndarray,
        labels: np.ndarray,
        output_path: str,
        raw_scores: Optional[np.ndarray] = None,
        n_bins: int = 10,
        title: str = "Reliability Diagram — CTR Predictor Calibration",
        figsize: Tuple[int, int] = (7, 7),
    ) -> str:
        """
        Plot a reliability (calibration) diagram.

        Each point represents one score bin.  x-axis = mean predicted
        probability in that bin; y-axis = actual positive fraction.
        A perfectly calibrated model falls exactly on the diagonal.
        The secondary y-axis shows sample counts per bin as a histogram.

        If *raw_scores* is provided, the uncalibrated curve is also plotted
        for direct comparison.

        Args:
            calibrated_scores:  ``(N,)`` calibrated probabilities.
            labels:             ``(N,)`` ground-truth values.
            output_path:        Destination PNG file.
            raw_scores:         Optional ``(N,)`` raw predictor scores for
                                before/after comparison.
            n_bins:             Number of calibration bins.
            title:              Figure title.
            figsize:            ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        threshold = float(np.median(labels))
        binary = (labels > threshold).astype(np.float32)
        ece = self.expected_calibration_error(calibrated_scores, labels, n_bins)

        def _calibration_curve(
            scores: np.ndarray, blab: np.ndarray, n_b: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            edges = np.linspace(0.0, 1.0, n_b + 1)
            idx = np.clip(np.digitize(scores, edges) - 1, 0, n_b - 1)
            mean_pred, mean_act, sizes = [], [], []
            for b in range(n_b):
                m = idx == b
                if not m.any():
                    continue
                mean_pred.append(float(scores[m].mean()))
                mean_act.append(float(blab[m].mean()))
                sizes.append(int(m.sum()))
            return np.array(mean_pred), np.array(mean_act), np.array(sizes)

        pred_cal, act_cal, sizes_cal = _calibration_curve(
            calibrated_scores, binary, n_bins
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Perfect-calibration diagonal
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.45,
            label="Perfect calibration",
        )

        # Uncalibrated overlay (optional)
        if raw_scores is not None:
            raw_range = float(raw_scores.max() - raw_scores.min())
            raw_norm = (
                (raw_scores - raw_scores.min()) / max(raw_range, 1e-8)
            ).astype(np.float32)
            pred_raw, act_raw, _ = _calibration_curve(raw_norm, binary, n_bins)
            ax.plot(
                pred_raw, act_raw, "o--",
                color="#e07b54", linewidth=1.5, markersize=5, alpha=0.75,
                label="Uncalibrated (normalised)",
            )

        # Calibrated curve
        ax.plot(
            pred_cal, act_cal, "o-",
            color="#5486e0", linewidth=2.2, markersize=7,
            label=f"Calibrated ({self.method})  ECE={ece:.3f}",
        )

        # Bin-size histogram on secondary axis
        ax2 = ax.twinx()
        ax2.bar(
            pred_cal, sizes_cal, width=0.04, alpha=0.18, color="#5486e0",
            align="center",
        )
        ax2.set_ylabel("Samples per bin", fontsize=8, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)

        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Actual positive fraction", fontsize=10)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12, pad=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(linestyle=":", alpha=0.35)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            "Reliability diagram saved to %s (ECE=%.4f).", output_path, ece
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration_params(self, output_path: str) -> str:
        """
        Serialise calibration parameters to JSON.

        Platt scaling coefficients (a, b) are saved for reproducibility.
        Isotonic regression is not directly JSON-serialisable; a summary
        is saved instead.

        Args:
            output_path:  Destination JSON file.

        Returns:
            Absolute path to the written file.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before save_calibration_params().")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        doc: Dict = {
            "method": self.method,
            "n_train_samples": self._n_train,
            "median_threshold": self._threshold,
        }
        if self.method == "platt" and hasattr(self._calibrator, "coef_"):
            doc["platt_a"] = float(self._calibrator.coef_[0][0])
            doc["platt_b"] = float(self._calibrator.intercept_[0])

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

        logger.info("Calibration params saved to %s.", output_path)
        return os.path.abspath(output_path)
