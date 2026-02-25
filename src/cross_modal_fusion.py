"""
cross_modal_fusion.py
---------------------
Uncertainty-weighted Bayesian fusion of audio and visual affective scores.

Motivation
~~~~~~~~~~
Simple concatenation or averaging of audio and visual affective vectors
treats all modalities as equally reliable, ignoring two critical factors:

1. **Modality reliability varies by creative type.**  A silent-cinema-style
   advert with no meaningful audio track has perfectly valid visual affective
   scores but near-random audio scores.  Averaging them naively degrades the
   visual signal.

2. **Per-frame ensemble confidence (from ``score_frames_ensemble``) quantifies
   how reproducible the visual affective probe is** — high variance across
   the 5-prompt ensemble means the frame embedding sits near the decision
   boundary of that axis.  This is a principled uncertainty signal.

Solution: **product-of-Gaussians (PoG) fusion** (Murphy 2007).
Given two independent Gaussian measurements of the same latent affective
value ``θ`` — one visual ``N(μ_v, σ_v²)`` and one audio ``N(μ_a, σ_a²)``
— the exact Bayesian posterior is also Gaussian:

    σ_fused² = 1 / (1/σ_v² + 1/σ_a²)                         [precision sum]
    μ_fused   = σ_fused² × (μ_v/σ_v² + μ_a/σ_a²)             [precision-weighted mean]

The confidence scores from ``score_frames_ensemble`` are mapped to
``σ_v²`` via ``σ_v² = 1 − confidence`` (low confidence → high variance).
Audio uncertainty is estimated from the spectral feature normalisation
margin (distance from the clipping threshold in ``audio_features.py``).

References
~~~~~~~~~~
- Murphy, K. P. (2007). *Conjugate Bayesian analysis of the Gaussian
  distribution.* Technical report, UBC.
- Brackett, D., & McLeod, I. M. (2000). *Pop music and the press.*
  Edinburgh University Press.  (Documents A/V discord effect on recall.)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# Minimum variance floor prevents division-by-zero when confidence = 1.0
_MIN_VARIANCE: float = 1e-4

# Default axis names (must match AffectiveScorer and AudioFeatureExtractor)
_AFFECTIVE_AXES: Tuple[str, ...] = (
    "energy", "warmth", "complexity", "luxury", "joy", "tension",
)

# When audio uncertainty is not provided, assume moderate reliability
_DEFAULT_AUDIO_VARIANCE: float = 0.25


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModalityFusionResult:
    """
    Fused affective scores for one frame or one video.

    Attributes:
        fused_scores:     Dict mapping axis name → fused point estimate.
        fused_variances:  Dict mapping axis name → posterior variance.
        visual_weight:    Dict mapping axis name → visual precision fraction ∈ [0, 1].
        audio_weight:     Dict mapping axis name → audio precision fraction ∈ [0, 1].
    """

    fused_scores: Dict[str, float]
    fused_variances: Dict[str, float]
    visual_weight: Dict[str, float]
    audio_weight: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------

def fuse_affective_scores(
    visual_scores: Dict[str, float],
    audio_scores: Dict[str, float],
    visual_variances: Optional[Dict[str, float]] = None,
    audio_variances: Optional[Dict[str, float]] = None,
    axes: Tuple[str, ...] = _AFFECTIVE_AXES,
) -> ModalityFusionResult:
    """
    Fuse visual and audio affective scores using product-of-Gaussians.

    Args:
        visual_scores:     Visual affective scores per axis (any range).
        audio_scores:      Audio affective scores per axis (any range).
        visual_variances:  Per-axis visual measurement variances.  When
                           *None*, defaults to ``_DEFAULT_AUDIO_VARIANCE``
                           for all axes.
        audio_variances:   Per-axis audio measurement variances.  When
                           *None*, defaults to ``_DEFAULT_AUDIO_VARIANCE``.
        axes:              Axis names to fuse; others are silently ignored.

    Returns:
        :class:`ModalityFusionResult` with precision-weighted fused scores
        and posterior variances.

    Raises:
        KeyError: if any axis is missing from *visual_scores* or
                  *audio_scores*.
    """
    fused_scores: Dict[str, float] = {}
    fused_variances: Dict[str, float] = {}
    visual_weight: Dict[str, float] = {}
    audio_weight: Dict[str, float] = {}

    for axis in axes:
        mu_v = float(visual_scores[axis])
        mu_a = float(audio_scores[axis])

        sig2_v = float(
            (visual_variances or {}).get(axis, _DEFAULT_AUDIO_VARIANCE)
        )
        sig2_a = float(
            (audio_variances or {}).get(axis, _DEFAULT_AUDIO_VARIANCE)
        )

        # Enforce minimum variance floor
        sig2_v = max(sig2_v, _MIN_VARIANCE)
        sig2_a = max(sig2_a, _MIN_VARIANCE)

        # Precision (inverse variance)
        prec_v = 1.0 / sig2_v
        prec_a = 1.0 / sig2_a
        prec_total = prec_v + prec_a

        # Posterior (product of Gaussians)
        sig2_fused = 1.0 / prec_total
        mu_fused = sig2_fused * (prec_v * mu_v + prec_a * mu_a)

        fused_scores[axis] = float(mu_fused)
        fused_variances[axis] = float(sig2_fused)
        visual_weight[axis] = float(prec_v / prec_total)
        audio_weight[axis] = float(prec_a / prec_total)

    return ModalityFusionResult(
        fused_scores=fused_scores,
        fused_variances=fused_variances,
        visual_weight=visual_weight,
        audio_weight=audio_weight,
    )


# ---------------------------------------------------------------------------
# Batch fusion
# ---------------------------------------------------------------------------

def fuse_frame_scores_batch(
    visual_scores_per_axis: Dict[str, np.ndarray],
    audio_scores: Dict[str, float],
    ensemble_confidence: Optional[Dict[str, np.ndarray]] = None,
    audio_variances: Optional[Dict[str, float]] = None,
    axes: Tuple[str, ...] = _AFFECTIVE_AXES,
) -> Dict[str, np.ndarray]:
    """
    Fuse per-frame visual scores with a single video-level audio score.

    When audio does not change frame-to-frame (typical for continuous audio),
    the audio signal provides a video-level prior, and the visual scores
    provide per-frame likelihoods.

    Args:
        visual_scores_per_axis:  Dict[axis → (N,) float32 array].
        audio_scores:            Video-level audio affective scores.
        ensemble_confidence:     Optional Dict[axis → (N,) float32 array]
                                 from ``score_frames_ensemble``.  Values in
                                 ``[0, 1]``; higher = more reliable.  Used
                                 to compute per-frame visual variances as
                                 ``σ_v² = 1 − confidence``.
        audio_variances:         Per-axis audio variances.
        axes:                    Axes to fuse.

    Returns:
        Dict[axis → (N,) float32 fused scores].

    Raises:
        ValueError: if any visual array is not 1-D.
        KeyError:   if any axis is missing.
    """
    # Validate shapes
    n_frames = None
    for axis in axes:
        arr = visual_scores_per_axis[axis]
        if arr.ndim != 1:
            raise ValueError(
                f"visual_scores_per_axis['{axis}'] must be 1-D; "
                f"got shape {arr.shape}."
            )
        if n_frames is None:
            n_frames = len(arr)
        elif len(arr) != n_frames:
            raise ValueError(
                f"Inconsistent lengths in visual_scores_per_axis: "
                f"first axis had {n_frames} frames but '{axis}' has {len(arr)}."
            )

    if n_frames is None or n_frames == 0:
        raise ValueError("visual_scores_per_axis contains no frames.")

    fused: Dict[str, np.ndarray] = {}

    for axis in axes:
        vis = visual_scores_per_axis[axis].astype(np.float64)
        aud_scalar = float(audio_scores[axis])
        aud_arr = np.full(n_frames, aud_scalar, dtype=np.float64)

        # Per-frame visual variance from ensemble confidence
        if ensemble_confidence is not None and axis in ensemble_confidence:
            conf = np.clip(ensemble_confidence[axis], 0.0, 1.0).astype(np.float64)
            sig2_v = np.maximum(1.0 - conf, _MIN_VARIANCE)
        else:
            sig2_v = np.full(n_frames, _DEFAULT_AUDIO_VARIANCE, dtype=np.float64)

        sig2_a = float(
            (audio_variances or {}).get(axis, _DEFAULT_AUDIO_VARIANCE)
        )
        sig2_a = max(sig2_a, _MIN_VARIANCE)

        prec_v = 1.0 / sig2_v
        prec_a = 1.0 / sig2_a
        prec_total = prec_v + prec_a

        fused_mu = (prec_v * vis + prec_a * aud_arr) / prec_total
        fused[axis] = fused_mu.astype(np.float32)

    return fused


# ---------------------------------------------------------------------------
# Variance estimation helpers
# ---------------------------------------------------------------------------

def ensemble_confidence_to_variances(
    confidence: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Convert ensemble confidence scores to per-frame measurement variances.

    Variance is defined as ``σ² = max(1 − confidence, _MIN_VARIANCE)`` so
    that high-confidence frames contribute more to any downstream fusion.

    Args:
        confidence:  Dict[axis → (N,) float32] from
                     ``AffectiveScorer.score_frames_ensemble``.

    Returns:
        Dict[axis → (N,) float32 variances].
    """
    return {
        axis: np.maximum(1.0 - np.clip(conf, 0.0, 1.0), _MIN_VARIANCE).astype(np.float32)
        for axis, conf in confidence.items()
    }


def audio_features_to_variances(
    audio_scores: Dict[str, float],
    clipping_threshold: float = 0.95,
) -> Dict[str, float]:
    """
    Estimate per-axis audio measurement variance from proximity to clipping.

    When a normalised feature value is near 1.0 (clipping threshold), the
    true underlying value is unknown (could be any value ≥ threshold).
    We model this as higher variance.

    ``σ²_audio = max(|score| / clipping_threshold, _MIN_VARIANCE)``

    Args:
        audio_scores:         Dict from ``map_to_affective_axes``.
        clipping_threshold:   Feature saturation threshold in ``[0, 1]``.

    Returns:
        Dict[axis → variance] values in ``[_MIN_VARIANCE, 1.0]``.
    """
    variances: Dict[str, float] = {}
    for axis, score in audio_scores.items():
        proximity = abs(score) / max(clipping_threshold, 1e-8)
        variances[axis] = float(np.clip(proximity, _MIN_VARIANCE, 1.0))
    return variances


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_fusion_weights(
    result: ModalityFusionResult,
    output_path: str,
    title: str = "Modality Fusion Weights per Affective Axis",
) -> None:
    """
    Plot a stacked bar chart showing visual vs audio weight per axis.

    Args:
        result:       :class:`ModalityFusionResult` for one frame/video.
        output_path:  Path to write the PNG.
        title:        Figure title.
    """
    axes = list(result.visual_weight.keys())
    vis_w = [result.visual_weight[a] for a in axes]
    aud_w = [result.audio_weight[a] for a in axes]

    x = np.arange(len(axes))
    fig, ax = plt.subplots(figsize=(max(6, len(axes) * 1.2), 4))

    bars_vis = ax.bar(x, vis_w, label="Visual", color="#4C72B0", alpha=0.85)
    bars_aud = ax.bar(x, aud_w, bottom=vis_w, label="Audio", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(axes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision weight")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Fusion weight chart saved: %s", output_path)


def plot_fused_vs_unimodal(
    visual_scores: Dict[str, float],
    audio_scores: Dict[str, float],
    fused_result: ModalityFusionResult,
    output_path: str,
    title: str = "Unimodal vs Fused Affective Scores",
) -> None:
    """
    Radar chart comparing visual-only, audio-only, and fused scores.

    Args:
        visual_scores:  Visual-only per-axis scores.
        audio_scores:   Audio-only per-axis scores.
        fused_result:   Fused result from :func:`fuse_affective_scores`.
        output_path:    Path to write the PNG.
        title:          Figure title.
    """
    axes_list = list(fused_result.fused_scores.keys())
    n = len(axes_list)
    if n < 3:
        logger.warning("plot_fused_vs_unimodal: fewer than 3 axes; skipping.")
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    def _to_radar(scores: Dict[str, float]) -> List[float]:
        vals = [float(scores.get(a, 0.0)) for a in axes_list]
        return vals + vals[:1]

    vis_vals = _to_radar(visual_scores)
    aud_vals = _to_radar(audio_scores)
    fused_vals = _to_radar(fused_result.fused_scores)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, vis_vals, "o-", linewidth=1.5, label="Visual", color="#4C72B0")
    ax.fill(angles, vis_vals, alpha=0.1, color="#4C72B0")
    ax.plot(angles, aud_vals, "s-", linewidth=1.5, label="Audio", color="#DD8452")
    ax.fill(angles, aud_vals, alpha=0.1, color="#DD8452")
    ax.plot(angles, fused_vals, "^-", linewidth=2, label="Fused", color="#55A868")
    ax.fill(angles, fused_vals, alpha=0.15, color="#55A868")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_list, size=9)
    ax.set_title(title, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Unimodal vs fused radar saved: %s", output_path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_fusion_results(
    results: List[ModalityFusionResult],
    index: List[Dict],
    output_path: str,
) -> None:
    """
    Save fusion results alongside frame metadata to a JSON file.

    Args:
        results:      One :class:`ModalityFusionResult` per frame.
        index:        Frame index list (same length as *results*).
        output_path:  Destination JSON path.

    Raises:
        ValueError: if ``len(results) != len(index)``.
    """
    if len(results) != len(index):
        raise ValueError(
            f"results ({len(results)}) and index ({len(index)}) must "
            "have the same length."
        )

    records = [
        {**meta, "fusion": result.to_dict()}
        for meta, result in zip(index, results)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
    logger.info("Fusion results saved: %d records → %s", len(records), output_path)
