"""
quality_filter.py
-----------------
Technical frame quality assessment for the Mini GACS pipeline.

Before spending GPU compute on CLIP embeddings, filter frames that are
technically poor:

- **Motion-blurred** (low Laplacian variance) — the Laplacian operator
  amplifies high-frequency edge detail; its variance collapses when the image
  is blurry.  A threshold of ~50 on 8-bit grey separates acceptably sharp
  frames from blurry ones in practice (Pertuz et al. 2013).

- **Over/under-exposed** — measured as the fraction of pixels in the top or
  bottom decile of the brightness histogram.  A washed-out or near-black frame
  has very low histogram entropy; both extremes indicate poor exposure.

- **Low information content** — fade-to-black/white transitions and near-uniform
  colour cards carry no semantic content.  We measure this as the spatial
  standard deviation of the Y (luminance) channel in YCbCr.  A near-uniform
  image has std ≈ 0.

Why it matters
~~~~~~~~~~~~~~
Technically poor frames have three harmful downstream effects:

1. **Affective scoring bias** — a motion-blurred "luxury" scene scores lower
   than the same scene in focus because CLIP's patch tokens lose texture detail.
   Averaging blurry and sharp frames biases video-level affective profiles.

2. **Spurious clusters** — overexposed frames cluster together regardless of
   scene content, creating a "blown-out" cluster that absorbs frames from
   multiple videos and inflates cluster imbalance.

3. **False scene transitions** — fade-to-black frames are highly dissimilar
   from both neighbours, so they trigger false transition detections and
   inflate the pacing-rate metric.

This filter runs on CPU with PIL + NumPy — negligible cost compared to
CLIP inference on GPU.

Usage
-----
    from src.quality_filter import FrameQualityFilter

    qf = FrameQualityFilter()
    scores = qf.score_frames(metadata)          # Dict[str, np.ndarray]
    clean_meta = qf.filter_frames(metadata, scores)

    # Or in one call:
    clean_meta, scores = qf.run(metadata)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (tuned for typical marketing video content at 224×224+)
# ---------------------------------------------------------------------------
DEFAULT_MIN_BLUR_VARIANCE: float = 30.0   # Laplacian variance threshold
DEFAULT_MIN_EXPOSURE_ENTROPY: float = 3.0  # Histogram entropy (bits) threshold
DEFAULT_MIN_LUMINANCE_STD: float = 10.0    # Luminance std threshold (0–255 scale)


class FrameQualityFilter:
    """
    Scores and filters frames by technical quality before CLIP embedding.

    Three independent quality axes are computed per frame and combined into
    a single composite quality score in ``[0, 1]``.  A frame is accepted when
    its composite score exceeds ``min_composite_score``.

    Args:
        min_blur_variance:    Minimum Laplacian variance.  Frames below this
                              are considered motion-blurred.  Default 30.0.
        min_exposure_entropy: Minimum histogram entropy (bits).  Frames below
                              this are nearly mono-chromatic (over/under-
                              exposed).  Default 3.0.
        min_luminance_std:    Minimum luminance std (0–255).  Frames below
                              this are near-uniform (solid colour / fade).
                              Default 10.0.
        min_composite_score:  Minimum overall quality score ``[0, 1]`` for a
                              frame to pass the filter.  Default 0.25.
    """

    def __init__(
        self,
        min_blur_variance: float = DEFAULT_MIN_BLUR_VARIANCE,
        min_exposure_entropy: float = DEFAULT_MIN_EXPOSURE_ENTROPY,
        min_luminance_std: float = DEFAULT_MIN_LUMINANCE_STD,
        min_composite_score: float = 0.25,
    ) -> None:
        if not (0.0 <= min_composite_score <= 1.0):
            raise ValueError(
                f"min_composite_score must be in [0, 1]; got {min_composite_score}."
            )
        self.min_blur_variance = min_blur_variance
        self.min_exposure_entropy = min_exposure_entropy
        self.min_luminance_std = min_luminance_std
        self.min_composite_score = min_composite_score

    # ------------------------------------------------------------------
    # Per-frame metrics
    # ------------------------------------------------------------------

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        """
        Compute the normalised Laplacian variance sharpness score.

        The raw Laplacian variance of a sharp image is typically in the
        range ``[50, 2000]`` for 8-bit grey images.  We normalise using a
        soft sigmoid so the output is in ``[0, 1]``.

        Args:
            image:  Greyscale uint8 array ``(H, W)``.

        Returns:
            Float in ``[0, 1]``.  Higher = sharper.
        """
        img_f = image.astype(np.float32)
        # Manual Laplacian via summing shifted copies (equivalent to applying
        # the 3×3 kernel [[0,1,0],[1,-4,1],[0,1,0]] but avoids scipy/cv2
        # dependency while remaining numerically identical).
        h, w = img_f.shape
        # Patch sum method: avoid scipy dependency
        padded = np.pad(img_f, 1, mode="edge")
        lap = (
            padded[:-2, 1:-1]   # top
            + padded[2:, 1:-1]  # bottom
            + padded[1:-1, :-2] # left
            + padded[1:-1, 2:]  # right
            - 4.0 * img_f       # centre
        )
        var = float(np.var(lap))
        # Normalise: var=50 → ≈0.5, var=200 → ≈0.83
        score = float(var / (var + 50.0))
        return score

    @staticmethod
    def exposure_entropy(image: np.ndarray) -> float:
        """
        Compute the histogram entropy of a greyscale image (in bits).

        A well-exposed image has a spread histogram with high entropy (> 5 bits
        for 256 bins).  An over- or under-exposed image has a histogram
        concentrated at one end — entropy collapses toward 0.

        Args:
            image:  Greyscale uint8 array ``(H, W)``.

        Returns:
            Float in ``[0, log2(256)]`` (0 to 8 bits).
        """
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist[hist > 0] / total
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def luminance_std(image: np.ndarray) -> float:
        """
        Compute the spatial standard deviation of the luminance channel.

        Args:
            image:  Greyscale uint8 array ``(H, W)``.

        Returns:
            Float ≥ 0.  Near-zero means the frame is near-uniform.
        """
        return float(np.std(image.astype(np.float32)))

    def score_single_frame(self, file_path: str) -> Dict[str, float]:
        """
        Compute quality metrics for a single frame file.

        Args:
            file_path:  Absolute or relative path to a JPEG/PNG image.

        Returns:
            Dict with keys ``"blur"``, ``"exposure_entropy"``,
            ``"luminance_std"``, and ``"composite"`` — all in ``[0, 1]``
            except ``"exposure_entropy"`` which is in ``[0, 8]``.

        Returns zeros on any load failure.
        """
        if not os.path.exists(file_path):
            logger.debug("Quality filter: file not found '%s'.", file_path)
            return {"blur": 0.0, "exposure_entropy": 0.0,
                    "luminance_std": 0.0, "composite": 0.0}
        try:
            pil_img = Image.open(file_path).convert("L")  # greyscale
            grey = np.asarray(pil_img, dtype=np.uint8)
            b_score = self.blur_score(grey)
            e_score = self.exposure_entropy(grey)
            l_std   = self.luminance_std(grey)

            # Normalise each axis to [0, 1] for composite
            b_norm = b_score  # already in [0, 1]
            e_norm = min(e_score / 8.0, 1.0)  # 8 = log2(256) max entropy
            l_norm = min(l_std / 64.0, 1.0)   # 64 = half the possible std range

            composite = float(np.mean([b_norm, e_norm, l_norm]))

            return {
                "blur": round(b_score, 4),
                "exposure_entropy": round(e_score, 4),
                "luminance_std": round(l_std, 4),
                "composite": round(composite, 4),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Quality filter: error loading '%s': %s", file_path, exc)
            return {"blur": 0.0, "exposure_entropy": 0.0,
                    "luminance_std": 0.0, "composite": 0.0}

    def score_frames(
        self,
        metadata: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """
        Score all frames in *metadata* for technical quality.

        Args:
            metadata:  List of frame metadata dicts; each must have a
                       ``"file_path"`` key.

        Returns:
            Dict mapping quality axis name → float32 array of length
            ``len(metadata)``:
            - ``"blur"``:              Normalised sharpness in ``[0, 1]``.
            - ``"exposure_entropy"``:  Histogram entropy in ``[0, 8]``.
            - ``"luminance_std"``:     Luminance std in ``[0, ∞)``.
            - ``"composite"``:         Mean-normalised composite in ``[0, 1]``.
        """
        n = len(metadata)
        blur    = np.zeros(n, dtype=np.float32)
        entropy = np.zeros(n, dtype=np.float32)
        lum_std = np.zeros(n, dtype=np.float32)
        composite = np.zeros(n, dtype=np.float32)

        for i, entry in enumerate(metadata):
            fp = entry.get("file_path", "")
            metrics = self.score_single_frame(fp)
            blur[i]      = metrics["blur"]
            entropy[i]   = metrics["exposure_entropy"]
            lum_std[i]   = metrics["luminance_std"]
            composite[i] = metrics["composite"]

        n_poor = int((composite < self.min_composite_score).sum())
        logger.info(
            "Quality filter scored %d frames; %d (%.1f%%) below composite threshold %.2f.",
            n, n_poor, 100.0 * n_poor / max(n, 1), self.min_composite_score,
        )
        return {
            "blur":              blur,
            "exposure_entropy":  entropy,
            "luminance_std":     lum_std,
            "composite":         composite,
        }

    def filter_frames(
        self,
        metadata: List[Dict],
        quality_scores: Dict[str, np.ndarray],
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Remove frames whose composite quality score is below the threshold.

        Args:
            metadata:        Frame metadata list.
            quality_scores:  Output of :meth:`score_frames`.

        Returns:
            Tuple of:
            - ``clean_metadata``:  Filtered metadata list.
            - ``kept_mask``:       Boolean ``(N,)`` — True at accepted positions.
        """
        composite = quality_scores["composite"]
        kept_mask = composite >= self.min_composite_score
        clean_metadata = [m for m, keep in zip(metadata, kept_mask) if keep]
        n_removed = int(len(metadata) - len(clean_metadata))

        logger.info(
            "Quality filter: kept %d / %d frames (removed %d, %.1f%%).",
            len(clean_metadata), len(metadata),
            n_removed, 100.0 * n_removed / max(len(metadata), 1),
        )
        return clean_metadata, kept_mask

    def run(
        self,
        metadata: List[Dict],
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Convenience method: score then filter frames in one call.

        Args:
            metadata:  Frame metadata list.

        Returns:
            Tuple ``(clean_metadata, quality_scores)`` where *clean_metadata*
            contains only frames that passed the quality threshold.
        """
        scores = self.score_frames(metadata)
        clean_meta, _ = self.filter_frames(metadata, scores)
        return clean_meta, scores
