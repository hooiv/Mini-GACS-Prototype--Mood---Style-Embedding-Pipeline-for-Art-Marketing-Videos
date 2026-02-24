"""
drift_detector.py
-----------------
Embedding distribution drift monitoring for the Mini GACS pipeline.

Problem
~~~~~~~
``VibePerformancePredictor`` is trained on one batch of creative embeddings.
In production, new batches of creatives are ingested continuously (new
campaigns, seasonal content, different cinematographers).  If the embedding
distribution of new creatives drifts significantly from the training
distribution, the predictor's feature-space assumptions may no longer hold,
leading to silently degraded predictions without any error signal.

This module provides a lightweight **drift detector** that compares a
"reference" embedding set (e.g. the batch used to train the predictor) to a
"new" set (the current pipeline run's embeddings), and reports:

1. **Maximum Mean Discrepancy (MMD)** — a kernel-based two-sample test
   statistic that is zero for identical distributions and positive otherwise.
   We use the RBF kernel with the standard median-heuristic bandwidth.

2. **Per-component Kolmogorov-Smirnov test** — projects both sets to the top
   ``n_components`` PCA dimensions (fitted on the reference set) and runs
   ``scipy.stats.ks_2samp`` on each component independently.  The minimum
   p-value across components is the most sensitive early-warning signal.

3. **Anomaly fraction** — the fraction of *new* frames flagged as outliers
   by an Isolation Forest fitted on the reference PCA projections.

4. **PCA scatter plot** — 2-D visualisation of the first two principal
   components, with reference frames in blue and new frames in orange.  A
   strong visual separation signals distribution shift.

Design note: all computations are performed in the reduced PCA space
(10 components by default), not in the original 512-D CLIP space.  This
serves two purposes:

a. **Speed**: MMD is O(N²) in the number of samples and O(D) in dimensionality.
   In 10-D (vs 512-D) the constant factor is 51× smaller.
b. **Stability**: concentration-of-measure effects in very high-dimensional
   spaces make Euclidean distances nearly uniform.  The top-10 PCA components
   capture 70–90 % of variance in typical CLIP embedding sets and live in a
   regime where distance statistics are well-behaved.

Usage
-----
    from src.drift_detector import EmbeddingDriftDetector

    detector = EmbeddingDriftDetector(n_components=10, alpha=0.05)
    detector.fit(reference_embeddings)                      # from training batch

    report = detector.detect(new_embeddings)
    print(report.mmd, report.is_drifted)

    detector.plot_pca_comparison(new_embeddings, "outputs/drift_pca.png")
    detector.save_report(report, "outputs/drift_report.json")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# The MMD kernel bandwidth is set using the median heuristic, which requires
# computing all pairwise squared distances in the PCA projection space.
# For N > MAX_MMD_SAMPLES we subsample to keep computation tractable.
_MAX_MMD_SAMPLES: int = 500

# Number of trees in the Isolation Forest.  100 is the sklearn default.
_IF_N_ESTIMATORS: int = 100

# Contamination parameter for IsolationForest: expected fraction of outliers
# in the reference set.  Set to "auto" (sklearn default ≈ 0.1) so the
# anomaly threshold adapts to the reference score distribution.
_IF_CONTAMINATION: str = "auto"


@dataclass
class DriftReport:
    """
    Summary of the drift analysis between a reference and a new embedding set.

    Fields
    ------
    mmd : float
        Maximum Mean Discrepancy (RBF kernel) between the PCA projections of
        reference and new embeddings.  Zero for identical distributions;
        larger values indicate greater distribution shift.
    ks_pvalue_min : float
        Minimum p-value across all per-component KS tests.  Small values
        (< ``alpha``) indicate at least one principal component has drifted.
    ks_pvalue_mean : float
        Mean p-value across all per-component KS tests.  A robust aggregate.
    is_drifted : bool
        ``True`` if ``ks_pvalue_min < alpha``.  The conservative flag that
        triggers a retraining or investigation alert.
    n_reference : int
        Number of frames in the reference set.
    n_new : int
        Number of frames in the new set.
    n_pca_components : int
        Number of PCA components used for all analyses.
    anomaly_fraction : float
        Fraction of *new* frames flagged as outliers by the Isolation Forest
        fitted on the reference set.  > 0.2 suggests meaningful OOD content.
    component_pvalues : list[float]
        Per-component KS p-values (length = ``n_pca_components``).
    """

    mmd: float
    ks_pvalue_min: float
    ks_pvalue_mean: float
    is_drifted: bool
    n_reference: int
    n_new: int
    n_pca_components: int
    anomaly_fraction: float
    component_pvalues: List[float]


class EmbeddingDriftDetector:
    """
    Detects distribution shift between a reference and a new embedding batch.

    Args:
        n_components:  Number of PCA components to use for analysis.
                       10 captures 70–90 % of variance in typical CLIP sets
                       while keeping MMD computation fast.
        alpha:         Significance level for the KS test.  Default 0.05.
        random_state:  Random seed for PCA and Isolation Forest reproducibility.
    """

    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be ≥ 1; got {n_components}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state

        self._pca: Optional[PCA] = None
        self._isolation_forest: Optional[IsolationForest] = None
        self._reference_projected: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, reference_embeddings: np.ndarray) -> "EmbeddingDriftDetector":
        """
        Fit the detector on a reference embedding set.

        Fits PCA and an Isolation Forest on the reference set.  All
        subsequent :meth:`detect` calls compare against this reference.

        Args:
            reference_embeddings:  L2-normalised float32 ``(N, D)`` array.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: if *reference_embeddings* is empty or not 2-D.
        """
        if reference_embeddings.ndim != 2 or reference_embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {reference_embeddings.shape}."
            )

        n, d = reference_embeddings.shape
        n_comp = min(self.n_components, n, d)

        self._pca = PCA(n_components=n_comp, random_state=self.random_state)
        self._reference_projected = self._pca.fit_transform(
            reference_embeddings.astype(np.float32)
        ).astype(np.float32)

        self._isolation_forest = IsolationForest(
            n_estimators=_IF_N_ESTIMATORS,
            contamination=_IF_CONTAMINATION,
            random_state=self.random_state,
        )
        self._isolation_forest.fit(self._reference_projected)

        self._is_fitted = True
        logger.info(
            "EmbeddingDriftDetector fitted: N_ref=%d, n_components=%d, "
            "explained_var_ratio=%.3f.",
            n, n_comp,
            float(self._pca.explained_variance_ratio_.sum()),
        )
        return self

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(self, new_embeddings: np.ndarray) -> DriftReport:
        """
        Compute a drift report comparing *new_embeddings* to the reference.

        Args:
            new_embeddings:  L2-normalised float32 ``(M, D)`` array.
                             Must have the same embedding dimension as the
                             reference set used in :meth:`fit`.

        Returns:
            :class:`DriftReport` with MMD, KS test results, anomaly fraction,
            and a boolean drift flag.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
            ValueError:   if *new_embeddings* is empty or wrong dimension.
        """
        if not self._is_fitted or self._pca is None:
            raise RuntimeError("Call fit() before detect().")

        if new_embeddings.ndim != 2 or new_embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {new_embeddings.shape}."
            )

        new_projected = self._pca.transform(
            new_embeddings.astype(np.float32)
        ).astype(np.float32)

        ref = self._reference_projected
        n_comp = ref.shape[1]

        # --- Per-component KS tests ---
        ks_pvalues: List[float] = []
        for c in range(n_comp):
            _, pval = ks_2samp(ref[:, c], new_projected[:, c])
            ks_pvalues.append(float(pval))

        ks_pvalue_min = float(min(ks_pvalues))
        ks_pvalue_mean = float(np.mean(ks_pvalues))
        is_drifted = ks_pvalue_min < self.alpha

        # --- MMD (RBF kernel, median bandwidth, capped subsampling) ---
        mmd = _compute_mmd_rbf(ref, new_projected, max_samples=_MAX_MMD_SAMPLES)

        # --- Anomaly fraction ---
        preds = self._isolation_forest.predict(new_projected)
        # IsolationForest returns -1 for outliers, +1 for inliers
        anomaly_fraction = float((preds == -1).mean())

        report = DriftReport(
            mmd=float(mmd),
            ks_pvalue_min=ks_pvalue_min,
            ks_pvalue_mean=ks_pvalue_mean,
            is_drifted=is_drifted,
            n_reference=int(ref.shape[0]),
            n_new=int(new_projected.shape[0]),
            n_pca_components=n_comp,
            anomaly_fraction=anomaly_fraction,
            component_pvalues=ks_pvalues,
        )
        logger.info(
            "Drift detection: MMD=%.4f, KS_min_pval=%.4f, drifted=%s, "
            "anomaly_frac=%.3f.",
            mmd, ks_pvalue_min, is_drifted, anomaly_fraction,
        )
        return report

    def is_distribution_drifted(self, new_embeddings: np.ndarray) -> bool:
        """
        Convenience method: returns ``True`` if drift is detected.

        Uses the same logic as :meth:`detect` but discards the full report.
        """
        return self.detect(new_embeddings).is_drifted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, report: DriftReport, output_path: str) -> str:
        """
        Save a :class:`DriftReport` to JSON.

        Args:
            report:       Output of :meth:`detect`.
            output_path:  Destination file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info(
            "Drift report saved to %s (drifted=%s).",
            output_path, report.is_drifted,
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_pca_comparison(
        self,
        new_embeddings: np.ndarray,
        output_path: str,
        title: str = "Embedding Distribution — Reference vs New Batch",
        figsize: Tuple[int, int] = (8, 6),
    ) -> str:
        """
        2-D PCA scatter: reference frames (blue) vs new frames (orange).

        Strong visual separation between the two clouds indicates meaningful
        distribution shift.  The percentage of variance explained by each
        component is shown on the axis labels.

        Args:
            new_embeddings:  L2-normalised ``(M, D)`` embeddings to compare.
            output_path:     Destination PNG file.
            title:           Figure title.
            figsize:         ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
        """
        if not self._is_fitted or self._pca is None:
            raise RuntimeError("Call fit() before plot_pca_comparison().")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        new_proj = self._pca.transform(
            new_embeddings.astype(np.float32)
        ).astype(np.float32)
        ref = self._reference_projected

        evr = self._pca.explained_variance_ratio_
        pc1_var = float(evr[0]) * 100 if len(evr) > 0 else 0.0
        pc2_var = float(evr[1]) * 100 if len(evr) > 1 else 0.0

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            ref[:, 0], ref[:, 1],
            c="#5486e0", alpha=0.55, s=18, label=f"Reference (N={len(ref)})",
            edgecolors="white", linewidths=0.3,
        )
        ax.scatter(
            new_proj[:, 0], new_proj[:, 1],
            c="#e07b54", alpha=0.65, s=22, label=f"New batch (N={len(new_proj)})",
            marker="^", edgecolors="white", linewidths=0.3,
        )
        ax.set_xlabel(f"PC 1 ({pc1_var:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC 2 ({pc2_var:.1f}% var)", fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(linestyle=":", alpha=0.35)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PCA comparison plot saved to %s.", output_path)
        return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# MMD implementation
# ---------------------------------------------------------------------------

def _compute_mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = _MAX_MMD_SAMPLES,
) -> float:
    """
    Compute the Maximum Mean Discrepancy with an RBF (Gaussian) kernel.

    The bandwidth parameter γ is set using the **median heuristic**:
    γ = 1 / (2 · median(||a − b||²)) over all pairs in the pooled sample.
    This is the standard adaptive bandwidth choice for MMD tests.

    The unbiased MMD² estimator is used::

        MMD²(X, Y) = (1/n(n-1)) Σ_{i≠j} k(x_i, x_j)
                   - (2/nm)      Σ_{i,j}  k(x_i, y_j)
                   + (1/m(m-1)) Σ_{i≠j} k(y_i, y_j)

    We return √max(MMD², 0) to correct for negative numerical artefacts in
    the unbiased estimator when the two distributions are very close.

    To keep complexity tractable we subsample to at most *max_samples* per
    set before computing kernel matrices.

    Args:
        X:            Reference PCA projections ``(n, k)``.
        Y:            New-batch PCA projections ``(m, k)``.
        max_samples:  Maximum samples from each set to include.

    Returns:
        Non-negative float: the estimated MMD distance.
    """
    rng = np.random.default_rng(42)

    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n, m = len(X), len(Y)

    # --- Median bandwidth heuristic ---
    XY = np.vstack([X, Y])
    # Squared pairwise distances via expansion: ||a-b||² = ||a||²+||b||²-2aᵀb
    sq_norms = np.sum(XY ** 2, axis=1)
    sq_dists = (
        sq_norms[:, None]
        + sq_norms[None, :]
        - 2.0 * (XY @ XY.T)
    )
    np.fill_diagonal(sq_dists, 0.0)  # numerical safety: ||a-a||² should be 0
    sq_dists = np.maximum(sq_dists, 0.0)  # numerical safety

    positive_dists = sq_dists[sq_dists > 1e-10]
    if positive_dists.size == 0:
        return 0.0  # all points identical — zero drift
    median_sq = float(np.median(positive_dists))
    gamma = 1.0 / (2.0 * max(median_sq, 1e-10))

    # --- Kernel matrices ---
    def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = (
            np.sum(A ** 2, axis=1)[:, None]
            + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        return np.exp(-gamma * np.maximum(sq, 0.0))

    Kxx = _rbf(X, X)
    Kyy = _rbf(Y, Y)
    Kxy = _rbf(X, Y)

    # Unbiased MMD² estimator (excludes diagonal in XX and YY)
    if n > 1:
        mmd2_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    else:
        mmd2_xx = 0.0
    if m > 1:
        mmd2_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    else:
        mmd2_yy = 0.0

    mmd2_xy = Kxy.mean()
    mmd2 = mmd2_xx - 2.0 * mmd2_xy + mmd2_yy

    return float(np.sqrt(max(mmd2, 0.0)))
