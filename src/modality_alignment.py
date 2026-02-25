"""
modality_alignment.py
---------------------
Corrects the CLIP modality gap documented in Liang et al. (2022)
"Mind the Gap: Understanding the Modality Gap in Multi-Modal Contrastive
Representation Learning", NeurIPS 2022.

The Problem
-----------
CLIP is trained with a contrastive loss that encourages matching image–text
pairs to have similar embeddings.  However, the training dynamics produce a
consistent *modality gap*: all image embeddings cluster in one cone of the
unit sphere, and all text embeddings cluster in a separate, non-overlapping
cone.  The gap vector ``g = mean(text) - mean(image)`` has a typical L2-norm
of 0.30–0.50 for ViT-B/32.

Effect on Affective Scoring
---------------------------
The affective axis score is::

    s_i = sim(image_i, text_pos) − sim(image_i, text_neg)

Because image embeddings live in a different cone from text embeddings, the
absolute cosine-similarity values are shifted toward the expected overlap
between cones (~0.20–0.30), *independently of actual semantic content*.  This
creates a **floor effect**: even a blank grey image scores near 0.0 on every
axis because the gap dominates the signal; only *relative* differences between
creatives are preserved, not their absolute magnitudes.  For a system that
aims to score "how energetic is this ad?", absolute magnitudes matter.

Correction
----------
Symmetric centering (adapted from Liang et al. §4.2)::

    corrected_image_i = L2_normalize( image_i  +  g / 2 )
    corrected_text_j  = L2_normalize( text_j   −  g / 2 )

where ``g`` is estimated from the empirical means of the current batch.
After correction both modalities live in the *same* cone, and cosine
similarities are dominated by semantic content rather than the gap.

The correction is purely additive — it does not require retraining CLIP or
access to the original contrastive training data.  It takes <1 ms for
typical creative library sizes.

Usage
-----
    from src.modality_alignment import ModalityAligner

    aligner = ModalityAligner()
    aligner.fit(frame_embeddings, text_anchor_embeddings)
    corrected_frames = aligner.correct_image(frame_embeddings)
    corrected_texts  = aligner.correct_text(text_anchor_embeddings)
    sims = aligner.cosine_similarity_corrected(frame_embeddings, text_anchors)

    aligner.save("outputs/modality_aligner.npz")
    loaded = ModalityAligner.load("outputs/modality_aligner.npz")
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Epsilon for numerically safe L2 normalisation
_L2_EPSILON: float = 1e-8


class ModalityAligner:
    """
    Estimates and corrects the CLIP modality gap.

    The gap vector is estimated as::

        gap = mean(text_embeddings, axis=0) − mean(image_embeddings, axis=0)

    Both sets of embeddings should be L2-normalised *before* fitting so that
    the gap reflects *direction* rather than scale.

    After :meth:`fit`, :meth:`correct_image` and :meth:`correct_text` apply
    the symmetric half-gap shift and re-normalise to the unit sphere.  The
    correction is idempotent when gap ≈ 0.

    Args:
        epsilon:  Small constant for numerically stable L2 normalisation.
    """

    def __init__(self, epsilon: float = _L2_EPSILON) -> None:
        self.epsilon = epsilon
        self._gap: Optional[np.ndarray] = None         # (D,)
        self._mean_image: Optional[np.ndarray] = None  # (D,)
        self._mean_text: Optional[np.ndarray] = None   # (D,)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has been called."""
        return self._gap is not None

    @property
    def gap(self) -> np.ndarray:
        """Gap vector ``mean(text) − mean(image)`` after :meth:`fit`."""
        if self._gap is None:
            raise RuntimeError("ModalityAligner has not been fitted yet.")
        return self._gap

    @property
    def gap_magnitude(self) -> float:
        """L2 norm of the gap vector.  Typical ViT-B/32 range: 0.30–0.50."""
        return float(np.linalg.norm(self.gap))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> "ModalityAligner":
        """
        Estimate the modality gap from image and text embedding sets.

        Args:
            image_embeddings:  L2-normalised image embeddings ``(N_img, D)``.
            text_embeddings:   L2-normalised text embeddings  ``(N_txt, D)``.

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: if arrays are not 2-D, are empty, or have mismatched
                        embedding dimensions.
        """
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("Both inputs must be 2-D arrays.")
        if image_embeddings.shape[0] == 0 or text_embeddings.shape[0] == 0:
            raise ValueError("Embedding arrays must be non-empty.")
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: "
                f"got {image_embeddings.shape[1]} vs {text_embeddings.shape[1]}."
            )

        self._mean_image = image_embeddings.mean(axis=0).astype(np.float32)
        self._mean_text  = text_embeddings.mean(axis=0).astype(np.float32)
        self._gap        = (self._mean_text - self._mean_image).astype(np.float32)

        logger.info(
            "ModalityAligner fitted: gap_magnitude=%.4f  (dim=%d, "
            "N_img=%d, N_txt=%d).",
            self.gap_magnitude,
            image_embeddings.shape[1],
            image_embeddings.shape[0],
            text_embeddings.shape[0],
        )
        return self

    def correct_image(self, image_embeddings: np.ndarray) -> np.ndarray:
        """
        Shift image embeddings by ``+gap/2`` and re-normalise.

        Args:
            image_embeddings:  ``(N, D)`` L2-normalised image embeddings.

        Returns:
            Gap-corrected, L2-normalised float32 array ``(N, D)``.

        Raises:
            RuntimeError: if the aligner has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before correct_image().")
        return self._l2_normalize(image_embeddings + self._gap / 2.0)

    def correct_text(self, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Shift text embeddings by ``-gap/2`` and re-normalise.

        Args:
            text_embeddings:  ``(N, D)`` L2-normalised text embeddings.

        Returns:
            Gap-corrected, L2-normalised float32 array ``(N, D)``.

        Raises:
            RuntimeError: if the aligner has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before correct_text().")
        return self._l2_normalize(text_embeddings - self._gap / 2.0)

    def cosine_similarity_corrected(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gap-corrected cosine similarities between images and texts.

        Corrects both modalities then computes the dot product (which equals
        cosine similarity for L2-normalised vectors).

        Args:
            image_embeddings:  ``(N_img, D)`` L2-normalised image embeddings.
            text_embeddings:   ``(N_txt, D)`` L2-normalised text embeddings.

        Returns:
            Float32 similarity matrix ``(N_img, N_txt)``.
        """
        corr_img  = self.correct_image(image_embeddings)
        corr_text = self.correct_text(text_embeddings)
        return (corr_img @ corr_text.T).astype(np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def gap_summary(self) -> dict:
        """
        Return a dict with gap diagnostics for logging and manifest.

        Keys: ``gap_magnitude``, ``mean_image_norm``, ``mean_text_norm``,
        ``embedding_dim``.
        """
        return {
            "gap_magnitude":  float(self.gap_magnitude),
            "mean_image_norm": float(np.linalg.norm(self._mean_image)),
            "mean_text_norm":  float(np.linalg.norm(self._mean_text)),
            "embedding_dim":   int(self._gap.shape[0]),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the gap vector and modality means to a compressed NumPy archive.

        Args:
            path:  Destination ``.npz`` file path.

        Raises:
            RuntimeError: if the aligner has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted ModalityAligner.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            gap=self._gap,
            mean_image=self._mean_image,
            mean_text=self._mean_text,
        )
        logger.info("ModalityAligner saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "ModalityAligner":
        """
        Load a previously saved aligner.

        Args:
            path:  Path to a ``.npz`` file written by :meth:`save`.

        Returns:
            A fitted :class:`ModalityAligner` instance.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Aligner file not found: '{path}'.")
        data = np.load(path)
        aligner = cls()
        aligner._gap        = data["gap"].astype(np.float32)
        aligner._mean_image = data["mean_image"].astype(np.float32)
        aligner._mean_text  = data["mean_text"].astype(np.float32)
        logger.info(
            "ModalityAligner loaded from '%s'  (gap_magnitude=%.4f).",
            path, aligner.gap_magnitude,
        )
        return aligner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation with epsilon guard."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return (x / (norms + self.epsilon)).astype(np.float32)
