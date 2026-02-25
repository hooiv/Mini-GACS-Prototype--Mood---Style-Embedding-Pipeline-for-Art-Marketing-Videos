"""
embedding_inspector.py
-----------------------
Intrinsic dimensionality estimation and embedding health diagnostics.

Why this matters
~~~~~~~~~~~~~~~~
CLIP produces 512-D embeddings, but real-world data typically lives on a
much lower-dimensional manifold.  Understanding the *intrinsic dimensionality*
(ID) and *geometric health* of your embedding set is critical because:

1. **Cluster selection bias** — K-means is biased when ID ≪ ambient D and
   the data has anisotropic spread.  `cluster_quality()` silhouette scores
   need context: a silhouette of 0.3 in ID=5 is good; the same in ID=50 is
   suspicious.

2. **Representation collapse** — Fine-tuned or quantised models sometimes
   collapse embeddings to a low-rank subspace (effective rank ≪ D).  This
   makes cosine similarity almost constant across all pairs, making the
   similarity matrix uninformative.

3. **Alignment quality** — CLIP's modality gap (Liang et al. 2022) shifts
   image embeddings away from text embeddings.  Anisotropy of the image
   embedding cloud tells us how severe this gap is and whether
   ``ModalityAligner`` is needed.

Metrics implemented
~~~~~~~~~~~~~~~~~~~
1. **TwoNN intrinsic dimensionality** (Facco et al. 2017, *Sci. Reports*).
   Estimates ID from the ratio of the 1st- and 2nd-nearest-neighbour
   distances.  Assumption: data is locally uniform on a d-dimensional
   manifold.  Robust to global non-linearities; O(N log N) with sklearn.

2. **Effective rank** (Roy & Vetterli 2007, *EUSIPCO*).
   ``eff_rank = exp(H(p))`` where ``H`` is the entropy of the normalised
   singular-value distribution.  Full-rank → ``eff_rank = D``; collapsed →
   ``eff_rank = 1``.

3. **Participation ratio** (alternative collapse metric).
   ``PR = (Σ λ_i)² / Σ λ_i²`` where λ are eigenvalues of the covariance.
   Same range as effective rank but more sensitive to extreme outliers.

4. **Anisotropy** — mean cosine similarity of the embedding matrix to its
   own mean direction.  Near 0 = isotropic (good); near 1 = collapsed to a
   ray (bad).

5. **NaN/Inf counts** and **L2-norm statistics** — basic sanity checks.

References
~~~~~~~~~~
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the
  intrinsic dimension of datasets by a minimal neighborhood information.
  *Scientific Reports*, 7, 12140.
- Roy, O., & Vetterli, M. (2007). The effective rank: A measure of effective
  dimensionality. *EUSIPCO 2007*.
- Liang, V. C. et al. (2022). Mind the gap: Understanding the modality gap
  in multi-modal contrastive representation learning. *NeurIPS 2022*.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# TwoNN: use exactly 2 neighbours per point; drop points where ratio == 1
# to avoid log(0) in the MLE.
_TWONN_N_NEIGHBORS: int = 2

# Effective rank: add this epsilon to eigenvalues before computing entropy
# to guard against numerical zero eigenvalues.
_EIGVAL_EPSILON: float = 1e-12

# Minimum number of embeddings required for meaningful diagnostics.
_MIN_SAMPLES_FOR_ID: int = 10


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingHealthReport:
    """
    Diagnostic report for a set of embeddings.

    Attributes:
        n_samples:            Number of embedding vectors.
        ambient_dim:          Raw dimensionality D (e.g. 512 for CLIP ViT-B/32).
        n_nan:                Number of embeddings containing NaN.
        n_inf:                Number of embeddings containing Inf.
        l2_norm_mean:         Mean L2 norm of all embeddings.
        l2_norm_std:          Std of L2 norms.
        intrinsic_dim_twonn:  TwoNN intrinsic dimensionality estimate.
        effective_rank:       Roy-Vetterli effective rank.
        participation_ratio:  Participation ratio of the covariance spectrum.
        anisotropy:           Mean cosine similarity to the mean direction.
        rank_fraction:        ``effective_rank / ambient_dim`` ∈ (0, 1].
        warnings:             List of human-readable diagnostic messages.
    """

    n_samples: int
    ambient_dim: int
    n_nan: int
    n_inf: int
    l2_norm_mean: float
    l2_norm_std: float
    intrinsic_dim_twonn: float
    effective_rank: float
    participation_ratio: float
    anisotropy: float
    rank_fraction: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"EmbeddingHealthReport ({self.n_samples} embeddings, D={self.ambient_dim})",
            f"  NaN / Inf counts   : {self.n_nan} / {self.n_inf}",
            f"  L2 norm            : {self.l2_norm_mean:.4f} ± {self.l2_norm_std:.4f}",
            f"  Intrinsic dim (TwoNN): {self.intrinsic_dim_twonn:.2f}",
            f"  Effective rank     : {self.effective_rank:.2f} / {self.ambient_dim}"
            f"  ({self.rank_fraction * 100:.1f}%)",
            f"  Participation ratio: {self.participation_ratio:.2f}",
            f"  Anisotropy         : {self.anisotropy:.4f}",
        ]
        if self.warnings:
            lines.append("  ⚠ Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TwoNN intrinsic dimensionality
# ---------------------------------------------------------------------------

def estimate_intrinsic_dim_twonn(embeddings: np.ndarray) -> float:
    """
    Estimate intrinsic dimensionality via the TwoNN estimator (Facco 2017).

    Algorithm:
        1. For each point, find its 1st and 2nd nearest neighbours.
        2. Compute ``μ_i = r2_i / r1_i`` (distance ratio).
        3. The ID estimate is ``1 / mean(log μ_i)``.

    Args:
        embeddings:  Float32 array ``(N, D)``.  Should be L2-normalised.

    Returns:
        Scalar float intrinsic dimension estimate.

    Raises:
        ValueError: if ``N < _MIN_SAMPLES_FOR_ID``.
    """
    n = embeddings.shape[0]
    if n < _MIN_SAMPLES_FOR_ID:
        raise ValueError(
            f"TwoNN requires at least {_MIN_SAMPLES_FOR_ID} samples; got {n}."
        )

    nn = NearestNeighbors(n_neighbors=_TWONN_N_NEIGHBORS + 1, algorithm="brute",
                          metric="euclidean")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # distances[:, 0] = distance to self (always 0); skip it
    r1 = distances[:, 1]   # 1st nearest neighbour distance
    r2 = distances[:, 2]   # 2nd nearest neighbour distance

    # Drop points where r1 == 0 (duplicates) to avoid log(0/0) = log(1) = 0
    valid = r1 > 0
    r1 = r1[valid]
    r2 = r2[valid]

    if len(r1) < 2:
        logger.warning("TwoNN: too many duplicate embeddings; ID estimate unreliable.")
        return float("nan")

    mu = r2 / np.maximum(r1, 1e-12)  # ratio ≥ 1 by triangle inequality
    # Discard degenerate ratios where r2 == r1 exactly
    mu = mu[mu > 1.0]

    if len(mu) < 2:
        logger.warning("TwoNN: too many equal r1/r2 pairs; returning NaN.")
        return float("nan")

    # MLE: − N / Σ log(μ_i)  (Facco Eq. 5)
    id_estimate = float(1.0 / np.mean(np.log(mu)))
    return max(id_estimate, 1.0)  # ID ≥ 1 by definition


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def compute_effective_rank(embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Compute Roy-Vetterli effective rank and participation ratio.

    Both metrics operate on the singular-value spectrum of the
    mean-centred embedding matrix.

    Args:
        embeddings:  Float32 array ``(N, D)``.

    Returns:
        Tuple ``(effective_rank, participation_ratio)``.
    """
    # Centre the embeddings
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)

    # Truncated SVD via covariance matrix (faster when N < D)
    if embeddings.shape[0] <= embeddings.shape[1]:
        cov = centred @ centred.T  # (N, N)
        eigvals = np.linalg.eigvalsh(cov)
    else:
        cov = centred.T @ centred  # (D, D)
        eigvals = np.linalg.eigvalsh(cov)

    eigvals = np.abs(eigvals)
    eigvals = eigvals + _EIGVAL_EPSILON
    total = eigvals.sum()

    # Effective rank: exp(Shannon entropy of normalised eigenvalue distribution)
    p = eigvals / total
    entropy = -np.sum(p * np.log(p))
    eff_rank = float(np.exp(entropy))

    # Participation ratio: (Σ λ)² / Σ λ²
    pr = float(total ** 2 / np.sum(eigvals ** 2))

    return eff_rank, pr


# ---------------------------------------------------------------------------
# Anisotropy
# ---------------------------------------------------------------------------

def compute_anisotropy(embeddings: np.ndarray) -> float:
    """
    Compute embedding cloud anisotropy as mean cosine similarity to the
    centroid direction.

    - Near 0 → isotropic (healthy, well-distributed).
    - Near 1 → all embeddings nearly parallel (collapsed to a ray).

    The CLIP modality gap manifests as moderate anisotropy (~0.3–0.5) in
    real-world embeddings.  After ``ModalityAligner`` correction this
    should decrease.

    Args:
        embeddings:  Float32 array ``(N, D)``.  **Must be L2-normalised.**

    Returns:
        Scalar float anisotropy value in ``[0, 1]``.
    """
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm < 1e-10:
        return 0.0

    centroid_dir = mean_emb / norm
    cosines = embeddings @ centroid_dir  # (N,)
    return float(np.clip(np.mean(cosines), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Full health report
# ---------------------------------------------------------------------------

def compute_health_report(
    embeddings: np.ndarray,
    l2_normalised: bool = True,
) -> EmbeddingHealthReport:
    """
    Compute a full embedding health report.

    Args:
        embeddings:     Float32 array ``(N, D)``.
        l2_normalised:  If *True*, skip L2-norm statistics (expected ≈ 1.0).

    Returns:
        :class:`EmbeddingHealthReport`.

    Raises:
        ValueError: if *embeddings* is not 2-D.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D; got shape {embeddings.shape}."
        )

    n, d = embeddings.shape
    warnings: List[str] = []

    # 1. Basic sanity
    n_nan = int(np.isnan(embeddings).sum())
    n_inf = int(np.isinf(embeddings).sum())
    if n_nan > 0:
        warnings.append(f"{n_nan} NaN value(s) detected in embeddings.")
    if n_inf > 0:
        warnings.append(f"{n_inf} Inf value(s) detected in embeddings.")

    # 2. L2 norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    l2_mean = float(norms.mean())
    l2_std = float(norms.std())
    if l2_normalised and l2_std > 0.01:
        warnings.append(
            f"Embeddings are not tightly L2-normalised: "
            f"mean={l2_mean:.4f}, std={l2_std:.4f}."
        )

    # Use clean embeddings for subsequent metrics
    clean_mask = np.isfinite(embeddings).all(axis=1)
    clean = embeddings[clean_mask]
    if len(clean) < _MIN_SAMPLES_FOR_ID:
        warnings.append(
            f"Only {len(clean)} clean embeddings; most metrics are unreliable."
        )
        return EmbeddingHealthReport(
            n_samples=n, ambient_dim=d, n_nan=n_nan, n_inf=n_inf,
            l2_norm_mean=l2_mean, l2_norm_std=l2_std,
            intrinsic_dim_twonn=float("nan"),
            effective_rank=float("nan"),
            participation_ratio=float("nan"),
            anisotropy=float("nan"),
            rank_fraction=float("nan"),
            warnings=warnings,
        )

    # L2-normalise clean embeddings for direction-based metrics
    norms_clean = np.linalg.norm(clean, axis=1, keepdims=True)
    unit = clean / np.maximum(norms_clean, 1e-12)

    # 3. TwoNN intrinsic dimensionality
    try:
        id_twonn = estimate_intrinsic_dim_twonn(unit)
    except ValueError:
        id_twonn = float("nan")
        warnings.append("TwoNN estimate failed (insufficient samples).")

    # 4. Effective rank + participation ratio
    try:
        eff_rank, pr = compute_effective_rank(clean)
    except np.linalg.LinAlgError:
        eff_rank = float("nan")
        pr = float("nan")
        warnings.append("Eigenvalue decomposition failed; effective rank unavailable.")

    # 5. Anisotropy
    anisotropy = compute_anisotropy(unit)
    if anisotropy > 0.7:
        warnings.append(
            f"High anisotropy ({anisotropy:.3f}) detected: embeddings may be "
            "collapsed toward a single direction. Consider running ModalityAligner."
        )

    # 6. Collapse warning based on effective rank
    rank_fraction = eff_rank / d if not np.isnan(eff_rank) else float("nan")
    if not np.isnan(rank_fraction) and rank_fraction < 0.05:
        warnings.append(
            f"Severe rank collapse: effective_rank={eff_rank:.1f} / {d} "
            f"({rank_fraction * 100:.1f}%). Cosine similarities are unreliable."
        )

    return EmbeddingHealthReport(
        n_samples=n,
        ambient_dim=d,
        n_nan=n_nan,
        n_inf=n_inf,
        l2_norm_mean=l2_mean,
        l2_norm_std=l2_std,
        intrinsic_dim_twonn=id_twonn,
        effective_rank=eff_rank,
        participation_ratio=pr,
        anisotropy=anisotropy,
        rank_fraction=rank_fraction,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_singular_value_spectrum(
    embeddings: np.ndarray,
    output_path: str,
    title: str = "Embedding Singular-Value Spectrum",
    top_k: int = 50,
) -> None:
    """
    Plot the normalised singular-value spectrum (scree plot) of the
    embedding matrix.

    A healthy, diverse embedding set shows a gradually declining spectrum.
    A collapsed set shows a sharp elbow at a small number of components.

    Args:
        embeddings:   Float32 array ``(N, D)``.
        output_path:  Path to write the PNG.
        title:        Figure title.
        top_k:        Number of singular values to show.
    """
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    k = min(top_k, min(centred.shape) - 1)

    # Use truncated SVD via randomised SVD for speed (sklearn)
    from sklearn.utils.extmath import randomized_svd
    _, sigma, _ = randomized_svd(centred, n_components=k, random_state=0)
    sigma_norm = sigma ** 2 / (sigma ** 2).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.arange(1, len(sigma_norm) + 1), sigma_norm, "o-",
                 markersize=4, linewidth=1.5)
    axes[0].set_xlabel("Component rank")
    axes[0].set_ylabel("Normalised eigenvalue")
    axes[0].set_title("Eigenvalue spectrum (scree plot)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, len(sigma_norm) + 1),
                 np.cumsum(sigma_norm) * 100, "s-", markersize=4, linewidth=1.5,
                 color="C1")
    axes[1].axhline(90, color="red", linestyle="--", linewidth=0.8,
                    label="90% variance")
    axes[1].set_xlabel("Component rank")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative variance explained")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Singular-value spectrum plot saved: %s", output_path)


def plot_health_dashboard(
    report: EmbeddingHealthReport,
    output_path: str,
) -> None:
    """
    Save a text-based health dashboard to a PNG (for pipeline manifests).

    Args:
        report:       :class:`EmbeddingHealthReport`.
        output_path:  Path to write the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    text = report.summary()
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox={"facecolor": "lightyellow", "alpha": 0.8, "pad": 10})

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Health dashboard saved: %s", output_path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_health_report(report: EmbeddingHealthReport, output_path: str) -> None:
    """
    Save :class:`EmbeddingHealthReport` as JSON.

    Args:
        report:       Report to save.
        output_path:  Destination ``.json`` file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(report.to_dict(), fp, indent=2)
    logger.info("Health report saved: %s", output_path)
