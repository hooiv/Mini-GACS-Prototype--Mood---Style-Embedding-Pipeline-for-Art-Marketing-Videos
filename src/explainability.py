"""
explainability.py
-----------------
Spatial explainability for CLIP-based affective scores via occlusion saliency.

Problem
~~~~~~~
``AffectiveScorer`` produces a single scalar for each axis (e.g., energy = 0.72)
but gives no indication of *where* in the image that score originates.  This is
the classic black-box problem: creative teams cannot act on "this frame scores low
on luxury" without knowing *which region* is driving the score.

Approach: Occlusion Saliency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Occlusion saliency (Zeiler & Fergus 2014, "Visualizing and Understanding
Convolutional Networks") systematically replaces patches of the input image with
a neutral fill (mid-grey, 128) and measures how much each replacement changes the
model output.

For affective axis scoring::

    baseline_score[axis]  = sim(full_frame_emb, pos_anchor)
                           − sim(full_frame_emb, neg_anchor)

    occluded_score[axis, i, j]  = sim(occluded_emb[i,j], pos_anchor)
                                  − sim(occluded_emb[i,j], neg_anchor)

    importance[axis, i, j] = baseline_score[axis]
                            − occluded_score[axis, i, j]

Interpretation::

    importance > 0  →  patch *increased* the axis score.  Removing it hurts.
    importance < 0  →  patch *suppressed* the axis score.  Removing it helps.
    importance ≈ 0  →  patch is irrelevant for this axis.

Why occlusion and not gradient-based saliency?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLIP ViT-B/32 uses a Vision Transformer.  Standard Grad-CAM requires gradients
of the class score w.r.t. final convolutional feature maps.  ViT has no spatial
feature map; the [CLS] token aggregates all patch tokens through 12 attention
layers.  Raw attention weights are not reliable importance proxies (Jain & Wallace
2019, "Attention is Not Explanation"), and proper gradient-based methods
(e.g. GradCAM++, Attention Rollout) require accessing multi-layer activations
with forward hooks, adding significant complexity.

Occlusion saliency:
- Is **gradient-free** — works identically regardless of architecture.
- Measures **causal** contribution (removing the patch = interventional do-operator).
- Costs exactly 1 extra batched forward pass (N_patches images in one GPU call).
- Produces interpretable spatial maps that creative directors can act on.

Performance
~~~~~~~~~~~
For a 4×4 grid the cost is 16 CLIP forward passes per frame, all batched into
a single GPU call.  For the ViT-B/32 model this takes ~20 ms on a modern GPU —
similar to 2–3 baseline forward passes.  Axis anchor embeddings are cached so
text-encoder overhead is paid only once per ``OcclusionSaliency`` instance.

Usage
-----
    from src.explainability import OcclusionSaliency

    occ = OcclusionSaliency(scorer, grid_rows=4, grid_cols=4)
    saliency = occ.compute_saliency(
        "data/frames/video_a_000.jpg",
        baseline_scores={"energy": 0.72, "warmth": 0.31},
    )
    # saliency: {"energy": (4, 4) array, "warmth": (4, 4) array}

    occ.plot_saliency_overlay(
        "data/frames/video_a_000.jpg",
        saliency,
        output_path="outputs/saliency_video_a_000.png",
    )

    top3 = occ.top_important_patches(saliency, "energy", top_k=3)
    # [(row, col, importance), ...]
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Grey fill value for occluded patches (per-channel, 0–255).
# 128 ≈ mid-grey; close to the mean pixel intensity of ImageNet images,
# which is what CLIP's vision encoder was pre-normalised against.
_OCCLUSION_FILL_VALUE: int = 128


class OcclusionSaliency:
    """
    Occlusion-based spatial saliency for CLIP affective axis scores.

    For each patch in a ``grid_rows × grid_cols`` grid, the patch is replaced
    with a uniform grey fill, the image is re-encoded by CLIP, and the change in
    each affective axis score is recorded as the patch's *importance* for that axis.

    All occluded images are batched into a single CLIP forward pass per
    :meth:`compute_saliency` call, keeping GPU overhead minimal.

    Axis anchor text embeddings are cached after the first call so text-encoder
    overhead is not repeated across multiple ``compute_saliency`` calls on the
    same instance.

    Args:
        scorer:      A configured :class:`src.affective_scoring.AffectiveScorer`.
                     Must share the same CLIP checkpoint as the embeddings used
                     to compute ``baseline_scores``.
        grid_rows:   Number of horizontal strips (default 4).
        grid_cols:   Number of vertical strips (default 4).
        fill_value:  Greyscale fill intensity for occluded patches (0–255).
                     Default 128 (mid-grey).
    """

    def __init__(
        self,
        scorer,
        grid_rows: int = 4,
        grid_cols: int = 4,
        fill_value: int = _OCCLUSION_FILL_VALUE,
    ) -> None:
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError(
                f"grid_rows and grid_cols must be ≥ 1; "
                f"got {grid_rows}×{grid_cols}."
            )
        self.scorer = scorer
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fill_value = fill_value
        # Lazy cache: populated on first call to _get_axis_anchors()
        self._axis_anchor_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    # ------------------------------------------------------------------
    # Axis anchor cache
    # ------------------------------------------------------------------

    def _get_axis_anchors(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return cached per-axis ``(pos_anchor, neg_anchor)`` text embeddings.

        Anchor embeddings are derived from the axis-definition text prompts in
        ``scorer.axes`` and are identical for every image.  Caching avoids
        re-running the CLIP text encoder on every :meth:`compute_saliency` call.

        Returns:
            Dict ``{axis_name: (pos_emb, neg_emb)}`` where each embedding is a
            float32 ``(D,)`` L2-normalised vector.
        """
        if self._axis_anchor_cache is None:
            self._axis_anchor_cache = {}
            for axis_name, (pos_prompt, neg_prompt) in self.scorer.axes.items():
                text_embs = self.scorer.encode_text([pos_prompt, neg_prompt])
                self._axis_anchor_cache[axis_name] = (text_embs[0], text_embs[1])
            logger.debug(
                "OcclusionSaliency: axis anchor cache populated (%d axes).",
                len(self._axis_anchor_cache),
            )
        return self._axis_anchor_cache

    def invalidate_cache(self) -> None:
        """
        Clear the cached axis anchor embeddings.

        Call this if ``scorer.axes`` is modified after the first
        :meth:`compute_saliency` call.
        """
        self._axis_anchor_cache = None
        logger.debug("OcclusionSaliency: axis anchor cache invalidated.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Compute L2-normalised CLIP image embeddings for a list of PIL images.

        All images are processed in a single batched forward pass via the
        scorer's processor and model.

        Args:
            images:  List of PIL ``Image`` objects in RGB mode.

        Returns:
            Float32 ``(N, D)`` L2-normalised array where N = len(images).
        """
        import torch

        inputs = self.scorer.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.scorer.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.scorer.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def _make_occluded_images(
        self,
        image: Image.Image,
    ) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
        """
        Generate all occluded variants of *image* for the configured grid.

        For a ``grid_rows × grid_cols`` grid this produces
        ``grid_rows × grid_cols`` images, each with one cell replaced by a
        uniform grey fill at intensity ``fill_value``.

        Args:
            image:  Source PIL image in RGB mode.

        Returns:
            Tuple ``(occluded_images, patch_positions)`` where each position
            is a ``(row, col)`` 0-indexed pair in row-major order.
        """
        w, h = image.size
        img_arr = np.array(image, dtype=np.uint8)  # (H, W, 3)
        occluded: List[Image.Image] = []
        positions: List[Tuple[int, int]] = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Integer-division boundaries; last cell absorbs remainder pixels
                y0 = row * h // self.grid_rows
                y1 = (row + 1) * h // self.grid_rows
                x0 = col * w // self.grid_cols
                x1 = (col + 1) * w // self.grid_cols

                patched = img_arr.copy()
                patched[y0:y1, x0:x1] = self.fill_value
                occluded.append(Image.fromarray(patched))
                positions.append((row, col))

        return occluded, positions

    # ------------------------------------------------------------------
    # Core saliency computation
    # ------------------------------------------------------------------

    def compute_saliency(
        self,
        image_path: str,
        baseline_scores: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-patch importance maps for each affective axis.

        Each patch in the ``grid_rows × grid_cols`` grid is occluded with
        a grey fill, the occluded image is re-encoded by CLIP, and the
        change in each affective axis score relative to ``baseline_scores``
        is recorded as that patch's importance.

        All occluded images are batched into a single CLIP forward pass.
        Axis anchor text embeddings are fetched from the cache (populated
        on the first call).

        Args:
            image_path:       Path to the image file (JPEG, PNG, …).
            baseline_scores:  Per-axis scores for the *unoccluded* image,
                              as returned by ``AffectiveScorer.score_frames``
                              for this single frame.  Format::

                                  {"energy": 0.72, "warmth": 0.31, ...}

        Returns:
            Dict ``{axis_name: (grid_rows, grid_cols) float32 array}`` of
            importance values.

            - Positive entry: the patch *increased* the axis score
              (removing it hurts the score).
            - Negative entry: the patch *suppressed* the axis score
              (removing it raises the score).
            - Near-zero: the patch is irrelevant for this axis.

        Raises:
            FileNotFoundError: if *image_path* does not exist.
            ValueError:        if *baseline_scores* is empty or contains NaN.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not baseline_scores:
            raise ValueError("baseline_scores must be a non-empty dict.")
        if any(
            isinstance(v, float) and np.isnan(v) for v in baseline_scores.values()
        ):
            raise ValueError("baseline_scores contains NaN values.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        # Generate and batch-encode all occluded variants
        occluded_images, _positions = self._make_occluded_images(image)
        occluded_embs = self._embed_images(occluded_images)  # (n_patches, D)

        # Fetch cached axis anchor embeddings
        anchors = self._get_axis_anchors()

        importance: Dict[str, np.ndarray] = {}
        n_patches = self.grid_rows * self.grid_cols

        for axis_name, (pos_anchor, neg_anchor) in anchors.items():
            if axis_name not in baseline_scores:
                # Axis not in baseline; skip silently
                continue

            baseline = float(baseline_scores[axis_name])

            # Axis score for each occluded image: (n_patches,)
            occ_scores = np.clip(
                occluded_embs @ pos_anchor - occluded_embs @ neg_anchor,
                -2.0, 2.0,
            ).astype(np.float32)

            # importance = how much this patch contributed to the baseline score
            # positive → patch raised the score; negative → patch lowered it
            delta = (baseline - occ_scores).astype(np.float32)  # (n_patches,)

            # Reshape from flat row-major order into (grid_rows, grid_cols)
            importance[axis_name] = delta.reshape(self.grid_rows, self.grid_cols)

        logger.info(
            "OcclusionSaliency: %d patches × %d axes computed for '%s'.",
            n_patches, len(importance), os.path.basename(image_path),
        )
        return importance

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def batch_compute_saliency(
        self,
        image_paths: List[str],
        baseline_scores_list: List[Dict[str, float]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Compute occlusion saliency for a list of frames.

        Processes frames sequentially.  Each frame's occluded patches are
        still batched into a single GPU forward pass per frame.

        Args:
            image_paths:           List of image file paths.
            baseline_scores_list:  Aligned list of per-axis score dicts
                                   (same order as *image_paths*).

        Returns:
            List of saliency dicts (same length as *image_paths*).  On
            error for an individual frame an empty dict is returned for
            that position and a warning is logged.

        Raises:
            ValueError: if the two lists have different lengths.
        """
        if len(image_paths) != len(baseline_scores_list):
            raise ValueError(
                "image_paths and baseline_scores_list must have the same length "
                f"(got {len(image_paths)} and {len(baseline_scores_list)})."
            )
        results: List[Dict[str, np.ndarray]] = []
        for path, scores in zip(image_paths, baseline_scores_list):
            try:
                results.append(self.compute_saliency(path, scores))
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Saliency skipped for '%s': %s", path, exc)
                results.append({})
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def top_important_patches(
        self,
        saliency: Dict[str, np.ndarray],
        axis_name: str,
        top_k: int = 3,
    ) -> List[Tuple[int, int, float]]:
        """
        Return the top-*k* most important (by absolute magnitude) patches
        for a given axis.

        Args:
            saliency:   Output of :meth:`compute_saliency`.
            axis_name:  Which axis to inspect.
            top_k:      How many patches to return.

        Returns:
            List of ``(row, col, importance)`` tuples sorted by descending
            absolute importance.

        Raises:
            KeyError: if *axis_name* is not in *saliency*.
        """
        if axis_name not in saliency:
            raise KeyError(
                f"Axis '{axis_name}' not found in saliency dict.  "
                f"Available axes: {sorted(saliency.keys())}."
            )
        grid = saliency[axis_name]
        n_patches = grid.size
        top_k = min(top_k, n_patches)
        flat_idx = np.argsort(np.abs(grid).ravel())[::-1][:top_k]
        rows, cols = np.unravel_index(flat_idx, grid.shape)
        return [
            (int(r), int(c), float(grid[r, c]))
            for r, c in zip(rows, cols)
        ]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_saliency_overlay(
        self,
        image_path: str,
        saliency: Dict[str, np.ndarray],
        output_path: str,
        figsize_per_panel: Tuple[int, int] = (4, 4),
        overlay_alpha: float = 0.50,
        cmap: str = "RdYlGn",
    ) -> str:
        """
        Multi-panel figure: original image + heatmap overlay for each axis.

        Each overlay super-imposes the importance heatmap — upsampled from the
        grid resolution to the full image dimensions using nearest-neighbour
        interpolation — onto the original image.  Green regions drive the score
        up; red regions drive it down.

        Args:
            image_path:        Path to the original image.
            saliency:          Output of :meth:`compute_saliency`.
            output_path:       Destination PNG file.
            figsize_per_panel: ``(width, height)`` in inches per panel.
            overlay_alpha:     Opacity of the heatmap layer (0 = invisible,
                               1 = fully opaque).
            cmap:              Matplotlib colormap (default ``"RdYlGn"``).

        Returns:
            Absolute path to the saved PNG.

        Raises:
            FileNotFoundError: if *image_path* does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        w, h = image.size
        n_axes = len(saliency)
        n_panels = 1 + n_axes  # original + one per axis

        fig, axes_arr = plt.subplots(
            1, n_panels,
            figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        )
        # Ensure axes_arr is always iterable even for 1 panel
        if n_panels == 1:
            axes_arr = [axes_arr]

        # Panel 0: original image
        axes_arr[0].imshow(image)
        axes_arr[0].set_title("Original", fontsize=10)
        axes_arr[0].axis("off")

        # Panels 1..n_axes: saliency overlay per axis
        for panel_idx, (axis_name, grid) in enumerate(saliency.items(), start=1):
            # Upsample importance grid to full image size (nearest-neighbour).
            # Min-max scale to [0, 255] uint8 for PIL:
            #   scaled = (grid − min) / (max − min + ε) × 255
            # The ε floor (1e-8) avoids division by zero for uniform grids.
            grid_range = max(float(grid.max() - grid.min()), 1e-8)
            grid_uint8 = ((grid - grid.min()) / grid_range * 255).astype(np.uint8)
            grid_pil = Image.fromarray(grid_uint8).resize((w, h), Image.NEAREST)
            grid_np = np.array(grid_pil)  # (H, W) uint8

            # Show original as base layer
            axes_arr[panel_idx].imshow(image)
            # Overlay saliency heatmap with transparency
            axes_arr[panel_idx].imshow(
                grid_np,
                cmap=cmap,
                vmin=0, vmax=255,
                alpha=overlay_alpha,
                extent=[0, w, h, 0],
            )
            axes_arr[panel_idx].set_title(axis_name, fontsize=10)
            axes_arr[panel_idx].axis("off")

        fig.suptitle(
            f"Occlusion Saliency — {os.path.basename(image_path)}",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            "Saliency overlay saved to %s (%d axes).", output_path, n_axes
        )
        return os.path.abspath(output_path)
