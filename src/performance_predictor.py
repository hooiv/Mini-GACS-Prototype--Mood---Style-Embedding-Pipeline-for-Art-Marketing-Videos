"""
performance_predictor.py
------------------------
Implements **REPORT.md §3A — Vibe–Performance Regression**.

This module provides:

1. **Synthetic performance data generator** — creates plausible CTR/ROAS
   figures from affective scores + embedding dimensions with injected noise,
   so the full R&D loop can be demonstrated without real campaign data.

2. **VibePerformancePredictor** — a ridge-regression head (with optional
   MLP fallback) that maps a frame/video's vibe feature vector to a
   predicted performance score.  Follows the scikit-learn estimator API.

3. **Cross-validation + Spearman ρ evaluation** — the gold-standard
   ranking-correlation metric for ad-scoring systems.

4. **Feature importance** — identifies which affective or embedding
   dimensions most strongly drive the predicted performance score,
   enabling actionable creative recommendations ("increase warmth").

Architecture
~~~~~~~~~~~~
::

    Vibe feature vector (embedding PCA + affective scores)
          │
          ▼
    StandardScaler ── Ridge / MLP ──► predicted CTR / ROAS
          │
          ▼
    Spearman ρ vs actual CTR  (cross-validated)
          │
          ▼
    Feature importance ──► "top positive drivers: warmth, joy"

Usage
-----
    from src.performance_predictor import (
        generate_synthetic_performance_data,
        VibePerformancePredictor,
    )

    # 1. Build features
    features, labels = generate_synthetic_performance_data(
        embeddings, affective_scores, index, target="ctr"
    )

    # 2. Train and cross-validate
    predictor = VibePerformancePredictor(model_type="ridge")
    cv_results = predictor.cross_validate(features, labels)

    # 3. Fit on full data and inspect importance
    predictor.fit(features, labels)
    importance = predictor.feature_importance(feature_names)

    # 4. Predict for new creatives
    predicted_ctr = predictor.predict(new_features)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_performance_data(
    embeddings: np.ndarray,
    affective_scores: Dict[str, np.ndarray],
    index: List[Dict],
    target: str = "ctr",
    noise_level: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CTR / ROAS labels by applying a *known* linear
    function of affective scores + top PCA components, then adding noise.

    This lets us verify the predictor can recover known signal — a standard
    R&D practice when real labels are unavailable.

    The ground-truth formula is::

        raw = w_joy * joy + w_energy * energy + w_warmth * warmth
              + w_pca1 * PCA1 + w_pca2 * PCA2 + noise

    For CTR the ground truth weights favour *joy* and *warmth* (positive
    emotional engagement).  For ROAS weights favour *luxury* and *complexity*.

    Args:
        embeddings:        L2-normalised ``(N, D)`` CLIP embeddings.
        affective_scores:  Dict ``{axis_name: (N,) array}`` from
                           :class:`src.affective_scoring.AffectiveScorer`.
        index:             Frame metadata list aligned with rows.
        target:            ``"ctr"`` or ``"roas"``.
        noise_level:       Standard deviation of Gaussian noise added to
                           the synthetic labels (relative to signal range).
        random_seed:       NumPy random seed.

    Returns:
        Tuple ``(features, labels)`` where:
        - *features* is float32 ``(N, F)`` combining affective scores and
          top-2 PCA components of the embeddings.
        - *labels* is float32 ``(N,)`` synthetic performance scores in
          approximately ``[0, 1]``.

    Raises:
        ValueError: if *embeddings* is empty or *target* is unknown.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Expected 2-D non-empty embeddings; got {embeddings.shape}.")
    if target not in ("ctr", "roas"):
        raise ValueError(f"Unknown target '{target}'; use 'ctr' or 'roas'.")

    n = embeddings.shape[0]
    rng = np.random.default_rng(random_seed)

    # --- PCA features from embeddings ---
    pca = PCA(n_components=min(5, embeddings.shape[1], n), random_state=random_seed)
    pca_features = pca.fit_transform(embeddings).astype(np.float32)

    # --- Affective features ---
    axis_names = list(affective_scores.keys())
    aff_matrix = np.column_stack([affective_scores[a] for a in axis_names]).astype(np.float32)
    # Normalise affective features to [0, 1] range for stable weighting
    aff_min = aff_matrix.min(axis=0, keepdims=True)
    aff_max = aff_matrix.max(axis=0, keepdims=True)
    aff_range = np.where(aff_max - aff_min > 1e-8, aff_max - aff_min, 1.0)
    aff_norm = (aff_matrix - aff_min) / aff_range

    # --- Combine features ---
    n_pca = pca_features.shape[1]
    features = np.hstack([aff_norm, pca_features]).astype(np.float32)

    # --- Build ground-truth weights ---
    axis_weights_ctr = {
        "joy": 0.6, "warmth": 0.5, "energy": 0.3,
        "luxury": 0.1, "complexity": -0.1, "tension": -0.3,
    }
    axis_weights_roas = {
        "luxury": 0.7, "complexity": 0.4, "warmth": 0.3,
        "energy": 0.2, "joy": 0.15, "tension": -0.2,
    }
    aw = axis_weights_ctr if target == "ctr" else axis_weights_roas

    w_aff = np.array([aw.get(a, 0.0) for a in axis_names], dtype=np.float32)
    # Small weight on PCA dimensions (random but fixed)
    rng_fixed = np.random.default_rng(0)
    w_pca = (rng_fixed.random(n_pca).astype(np.float32) - 0.5) * 0.2

    w = np.concatenate([w_aff, w_pca])

    # --- Compute signal + noise ---
    signal = features @ w
    noise_std = noise_level * (signal.max() - signal.min() + 1e-8)
    noise = rng.normal(0, noise_std, size=n).astype(np.float32)
    raw = signal + noise

    # Sigmoid to [0, 1]
    labels = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

    logger.info(
        "Synthetic %s data: N=%d, features=%d, label range=[%.4f, %.4f].",
        target.upper(), n, features.shape[1], float(labels.min()), float(labels.max()),
    )
    return features, labels


def build_feature_names(
    affective_scores: Dict[str, np.ndarray],
    n_pca: int,
) -> List[str]:
    """Return a list of feature names matching :func:`generate_synthetic_performance_data`."""
    return list(affective_scores.keys()) + [f"pca_{i}" for i in range(n_pca)]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class VibePerformancePredictor:
    """
    Predicts creative performance (CTR / ROAS) from vibe feature vectors.

    Wraps a scikit-learn pipeline:  ``StandardScaler → Ridge | MLP``.

    Args:
        model_type:    ``"ridge"`` (default, interpretable, fast) or
                       ``"mlp"`` (non-linear, higher capacity).
        alpha:         Ridge regularisation strength (used when
                       ``model_type="ridge"``).
        hidden_layers: Hidden layer sizes for MLP (used when
                       ``model_type="mlp"``).
        random_state:  Random seed.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        hidden_layers: Tuple[int, ...] = (64, 32),
        random_state: int = 42,
    ) -> None:
        if model_type not in ("ridge", "mlp"):
            raise ValueError(f"Unknown model_type '{model_type}'; use 'ridge' or 'mlp'.")
        self.model_type = model_type
        self.alpha = alpha
        self.hidden_layers = hidden_layers
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False
        self._feature_names: Optional[List[str]] = None

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "ridge":
            regressor = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            regressor = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation="relu",
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "VibePerformancePredictor":
        """
        Fit the predictor on *features* and *labels*.

        Args:
            features:       ``(N, F)`` float32 feature matrix.
            labels:         ``(N,)`` float32 target values.
            feature_names:  Optional list of *F* feature name strings for
                            interpretability.

        Returns:
            Self (for chaining).
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(features, labels)
        self._is_fitted = True
        self._feature_names = feature_names
        logger.info(
            "VibePerformancePredictor (%s) fitted on %d samples, %d features.",
            self.model_type, features.shape[0], features.shape[1],
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict performance scores for *features*.

        Args:
            features:  ``(M, F)`` float32 feature matrix.

        Returns:
            Float32 array of shape ``(M,)``.

        Raises:
            RuntimeError: if the predictor has not been fitted.
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        return self._pipeline.predict(features).astype(np.float32)

    def cross_validate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        K-fold cross-validation, reporting Spearman ρ and RMSE per fold.

        Spearman ρ is the canonical evaluation metric for creative scoring
        systems because it measures rank correlation — we care about whether
        we can *rank* creatives by predicted CTR, not whether the absolute
        numbers are accurate.

        Args:
            features:  ``(N, F)`` feature matrix.
            labels:    ``(N,)`` target values.
            n_splits:  Number of CV folds.

        Returns:
            Dict with ``mean_spearman``, ``std_spearman``, ``mean_rmse``,
            ``std_rmse``, and ``fold_spearman`` (list of per-fold ρ values).
        """
        n_splits = min(n_splits, len(labels))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        spearman_scores: List[float] = []
        rmse_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            X_tr, X_val = features[train_idx], features[val_idx]
            y_tr, y_val = labels[train_idx], labels[val_idx]

            pipeline = self._build_pipeline()
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val).astype(np.float32)

            rho, _ = spearmanr(y_val, y_pred)
            rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

            spearman_scores.append(float(rho) if not np.isnan(rho) else 0.0)
            rmse_scores.append(rmse)
            logger.debug("Fold %d: Spearman ρ=%.4f, RMSE=%.4f.", fold_idx, rho, rmse)

        results = {
            "mean_spearman": float(np.mean(spearman_scores)),
            "std_spearman": float(np.std(spearman_scores)),
            "mean_rmse": float(np.mean(rmse_scores)),
            "std_rmse": float(np.std(rmse_scores)),
            "fold_spearman": spearman_scores,
        }
        logger.info(
            "CV results (%d folds): Spearman ρ=%.4f ± %.4f, RMSE=%.4f ± %.4f.",
            n_splits,
            results["mean_spearman"], results["std_spearman"],
            results["mean_rmse"], results["std_rmse"],
        )
        return results

    def feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-*k* most influential features by absolute weight.

        For the ridge model the coefficients are directly interpretable.
        For the MLP model we use the L1-norm of the first-layer weights as
        a proxy importance (not causal, but a useful heuristic).

        Args:
            feature_names:  Override names for the feature columns.
            top_k:          How many top features to return.

        Returns:
            List of ``(feature_name, importance_score)`` tuples sorted by
            descending absolute importance.

        Raises:
            RuntimeError: if the predictor has not been fitted.
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Call fit() before feature_importance().")

        names = feature_names or self._feature_names

        regressor = self._pipeline.named_steps["regressor"]
        scaler: StandardScaler = self._pipeline.named_steps["scaler"]

        if self.model_type == "ridge":
            raw_coef = regressor.coef_  # (F,)
            # Scale coefficients by feature std to get comparable magnitudes
            std = np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
            importance = np.abs(raw_coef / std)
        else:
            # MLP: L1 norm of first-layer weight matrix columns → (F,)
            w1 = regressor.coefs_[0]  # (F, hidden[0])
            importance = np.abs(w1).sum(axis=1)

        n_feat = len(importance)
        if names is None or len(names) != n_feat:
            names = [f"feature_{i}" for i in range(n_feat)]

        ranked = sorted(
            zip(names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def save_model(self, output_path: str) -> str:
        """
        Serialise the fitted model to a JSON-friendly dict.

        Only ridge models can be fully serialised this way.  MLP models
        save a summary (not reloadable).

        Returns:
            Absolute path to the written file.
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Call fit() before save_model().")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        regressor = self._pipeline.named_steps["regressor"]
        scaler: StandardScaler = self._pipeline.named_steps["scaler"]

        doc: Dict = {
            "model_type": self.model_type,
            "alpha": self.alpha,
            "feature_names": self._feature_names,
            "scaler_mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
            "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        }
        if self.model_type == "ridge":
            doc["coef"] = regressor.coef_.tolist()
            doc["intercept"] = float(regressor.intercept_)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

        logger.info("Model summary saved to %s.", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_cv_results(
        self,
        cv_results: Dict,
        output_path: str,
        title: str = "Cross-Validation: Spearman ρ per Fold",
        figsize: Tuple[int, int] = (8, 4),
    ) -> str:
        """
        Bar chart of per-fold Spearman ρ with mean ± std annotation.

        Args:
            cv_results:   Output of :meth:`cross_validate`.
            output_path:  Destination PNG file.
            title:        Figure title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fold_rho = cv_results["fold_spearman"]
        mean_rho = cv_results["mean_spearman"]
        std_rho = cv_results["std_spearman"]

        fig, ax = plt.subplots(figsize=figsize)
        colours = ["#5486e0" if r >= 0 else "#e07b54" for r in fold_rho]
        ax.bar(range(len(fold_rho)), fold_rho, color=colours, alpha=0.8, edgecolor="white")
        ax.axhline(mean_rho, color="black", linewidth=1.5, linestyle="--",
                   label=f"mean ρ = {mean_rho:.3f} ± {std_rho:.3f}")
        ax.axhspan(mean_rho - std_rho, mean_rho + std_rho,
                   alpha=0.1, color="black")
        ax.set_xticks(range(len(fold_rho)))
        ax.set_xticklabels([f"Fold {i}" for i in range(len(fold_rho))], fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("CV results chart saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_feature_importance(
        self,
        output_path: str,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
        title: str = "Top Feature Importances (Vibe → Performance)",
        figsize: Tuple[int, int] = (8, 5),
    ) -> str:
        """
        Horizontal bar chart of top-k feature importances.

        Args:
            output_path:    Destination PNG.
            feature_names:  Override feature names.
            top_k:          Number of features shown.
            title:          Figure title.
            figsize:        ``(width, height)`` in inches.

        Returns:
            Absolute path to saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        importance_list = self.feature_importance(feature_names=feature_names, top_k=top_k)
        names = [n for n, _ in importance_list]
        values = [v for _, v in importance_list]

        fig, ax = plt.subplots(figsize=figsize)
        colours = ["#5486e0"] * len(names)
        # Highlight known affective axes in a different colour
        affective_axes = {"joy", "warmth", "energy", "luxury", "complexity", "tension"}
        colours = ["#e07b54" if n in affective_axes else "#5486e0" for n in names]

        y_pos = range(len(names))
        ax.barh(list(y_pos), values, color=colours, alpha=0.85, edgecolor="white")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance (|coefficient × 1/std|)", fontsize=9)
        ax.set_title(title, fontsize=12, pad=10)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e07b54", label="Affective axis"),
            Patch(facecolor="#5486e0", label="PCA dimension"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Feature importance chart saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_predicted_vs_actual(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        output_path: str,
        title: str = "Predicted vs Actual Performance (CTR/ROAS)",
        figsize: Tuple[int, int] = (7, 7),
    ) -> str:
        """
        Scatter plot of predicted vs actual performance scores.

        Args:
            actual:       ``(N,)`` ground-truth values.
            predicted:    ``(N,)`` model predictions.
            output_path:  Destination PNG.
            title:        Figure title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rho, _ = spearmanr(actual, predicted)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(actual, predicted, alpha=0.6, s=25, color="#5486e0",
                   edgecolors="white", linewidths=0.3)

        # Identity line
        lo = min(actual.min(), predicted.min()) - 0.02
        hi = max(actual.max(), predicted.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="ideal")

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            "Predicted vs actual plot saved to %s (Spearman ρ=%.4f).",
            output_path, rho,
        )
        return os.path.abspath(output_path)
