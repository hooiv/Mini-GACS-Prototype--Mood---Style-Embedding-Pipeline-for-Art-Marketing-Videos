"""
affective_scoring.py
--------------------
Zero-shot text-guided affective scoring using CLIP.

The key idea (described in REPORT.md §2a) is that CLIP's shared image-text
embedding space lets us probe a frame against descriptive text prompts without
any labelled data.  For each named "affective axis" (e.g. *energy*, *warmth*,
*complexity*) we define a positive and a negative anchor prompt and project the
frame embedding onto the axis direction:

    axis_score = sim(frame, positive_prompt) - sim(frame, negative_prompt)

The result is a scalar in ``[-1, 1]`` where +1 means the frame strongly
matches the positive pole and -1 means it strongly matches the negative pole.

Usage
-----
    from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

    scorer = AffectiveScorer()
    scores = scorer.score_frames(frame_embeddings, index)
    # scores: Dict[str, np.ndarray]  axis_name → (N,) array of floats

    scorer.save_scores(scores, index, "outputs/affective_scores.json")
    scorer.plot_radar(scores, index, "outputs/affective_radar.png")
    scorer.plot_heatmap(scores, index, "outputs/affective_heatmap.png")
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default affective axis definitions
# Each axis is a (positive_prompt, negative_prompt) pair.
# ---------------------------------------------------------------------------
DEFAULT_AXES: Dict[str, Tuple[str, str]] = {
    "energy":     ("an energetic, dynamic, fast-paced scene",
                   "a calm, still, peaceful scene"),
    "warmth":     ("a warm, golden, sun-lit scene with warm colours",
                   "a cold, blue-toned, icy scene"),
    "complexity": ("a visually complex, highly detailed, busy scene",
                   "a minimalist, clean, simple scene"),
    "luxury":     ("an opulent, high-end, luxurious aesthetic",
                   "a raw, gritty, low-budget aesthetic"),
    "joy":        ("a joyful, happy, uplifting scene",
                   "a dark, melancholic, sad scene"),
    "tension":    ("a tense, dramatic, suspenseful scene",
                   "a relaxed, serene, tension-free scene"),
}


class AffectiveScorer:
    """
    Scores image frame embeddings against named affective axes using CLIP
    text embeddings as axis anchors.

    Args:
        model_name:   HuggingFace CLIP identifier.  Must match the model used
                      to compute the frame embeddings.
        axes:         Affective-axis definitions.  Defaults to
                      :data:`DEFAULT_AXES`.
        device:       Torch device (auto-detected when *None*).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        axes: Optional[Dict[str, Tuple[str, str]]] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.axes = axes or DEFAULT_AXES

        logger.info(
            "Loading CLIP model '%s' for affective scoring on device '%s'.",
            model_name, device,
        )
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("AffectiveScorer ready with %d axes.", len(self.axes))

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def encode_text(self, prompts: List[str]) -> np.ndarray:
        """
        Compute L2-normalised CLIP text embeddings for a list of prompts.

        Args:
            prompts:  List of text strings.

        Returns:
            Float32 array ``(len(prompts), D)``.
        """
        inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def score_frames(
        self,
        frame_embeddings: np.ndarray,
        index: Optional[List[Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute affective axis scores for every frame.

        For each axis ``(pos_prompt, neg_prompt)`` the score is::

            score_i = sim(frame_i, pos_emb) - sim(frame_i, neg_emb)

        where ``sim`` is the dot product (equivalent to cosine similarity
        for L2-normalised vectors).

        Args:
            frame_embeddings:  Float32 ``(N, D)`` L2-normalised image embeddings.
            index:             Optional metadata list (not used for scoring;
                               retained for convenience when wiring with other
                               modules).

        Returns:
            Dict mapping axis name → float32 array of shape ``(N,)`` with
            values in ``[-1, 1]``.

        Raises:
            ValueError: if ``frame_embeddings`` is not 2-D or is empty.
        """
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {frame_embeddings.shape}."
            )

        axis_scores: Dict[str, np.ndarray] = {}

        for axis_name, (pos_prompt, neg_prompt) in self.axes.items():
            text_embs = self.encode_text([pos_prompt, neg_prompt])
            pos_emb = text_embs[0]   # (D,)
            neg_emb = text_embs[1]   # (D,)

            # Dot product = cosine sim for L2-normalised vectors.
            # Each sim value is in [-1, 1], so their difference spans [-2, 2].
            pos_sims = (frame_embeddings @ pos_emb).astype(np.float32)
            neg_sims = (frame_embeddings @ neg_emb).astype(np.float32)
            # Guard against floating-point overflow; natural range is [-2, 2]
            scores = np.clip(pos_sims - neg_sims, -2.0, 2.0)

            axis_scores[axis_name] = scores
            logger.debug(
                "Axis '%s': mean=%.4f, std=%.4f, range=[%.4f, %.4f].",
                axis_name,
                float(scores.mean()), float(scores.std()),
                float(scores.min()), float(scores.max()),
            )

        logger.info(
            "Affective scores computed for %d frames across %d axes.",
            frame_embeddings.shape[0], len(axis_scores),
        )
        return axis_scores

    def score_video_level(
        self,
        frame_scores: Dict[str, np.ndarray],
        index: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate frame-level axis scores to the video level (mean pooling).

        Args:
            frame_scores:  Output of :meth:`score_frames`.
            index:         Metadata list with ``"video_id"`` keys, aligned
                           with the rows of ``frame_scores`` values.

        Returns:
            Dict ``{video_id: {axis_name: mean_score, ...}, ...}``.
        """
        video_ids = [entry.get("video_id", "unknown") for entry in index]
        unique_videos = sorted(set(video_ids))

        video_scores: Dict[str, Dict[str, float]] = {v: {} for v in unique_videos}
        for axis_name, scores in frame_scores.items():
            for vid in unique_videos:
                mask = np.array([v == vid for v in video_ids])
                if mask.any():
                    video_scores[vid][axis_name] = float(scores[mask].mean())

        return video_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scores(
        self,
        frame_scores: Dict[str, np.ndarray],
        index: List[Dict],
        output_path: str,
    ) -> str:
        """
        Save frame-level affective scores to a JSON file.

        The format is a list of dicts, one per frame, containing all metadata
        fields from *index* plus one key per axis::

            [{"video_id": ..., "frame_idx": ..., "energy": 0.12, ...}, ...]

        Args:
            frame_scores:  Output of :meth:`score_frames`.
            index:         Aligned metadata list.
            output_path:   Destination JSON file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rows = []
        for i, entry in enumerate(index):
            row = dict(entry)
            for axis_name, scores in frame_scores.items():
                if i < len(scores):
                    row[axis_name] = round(float(scores[i]), 6)
            rows.append(row)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

        logger.info("Affective scores saved to %s (%d rows).", output_path, len(rows))
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        frame_scores: Dict[str, np.ndarray],
        index: List[Dict],
        output_path: str,
        title: str = "Frame-Level Affective Scores",
        figsize: Tuple[int, int] = (14, 6),
        max_labels: int = 40,
    ) -> str:
        """
        Plot a heatmap of affective scores (axes × frames).

        Args:
            frame_scores:  Dict of axis_name → (N,) score array.
            index:         Metadata list aligned with columns.
            output_path:   Destination PNG file.
            title:         Figure title.
            figsize:       ``(width, height)`` in inches.
            max_labels:    Max number of x-axis tick labels.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        axis_names = list(frame_scores.keys())
        matrix = np.vstack([frame_scores[ax] for ax in axis_names])  # (A, N)

        n = matrix.shape[1]
        x_labels = [
            f"{e.get('video_id','?')}/{e.get('frame_idx', i)}"
            for i, e in enumerate(index)
        ]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
        plt.colorbar(im, ax=ax, label="Affective Score (pos − neg)")

        ax.set_yticks(range(len(axis_names)))
        ax.set_yticklabels(axis_names, fontsize=10)

        if n <= max_labels:
            ax.set_xticks(range(n))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"{n} frames (tick labels hidden for clarity)")

        ax.set_title(title, fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Affective score heatmap saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_radar(
        self,
        video_scores: Dict[str, Dict[str, float]],
        output_path: str,
        title: str = "Video-Level Affective Profile",
        figsize: Tuple[int, int] = (8, 8),
    ) -> str:
        """
        Plot a radar (spider) chart comparing the affective profiles of
        different videos.

        Args:
            video_scores:  Output of :meth:`score_video_level`.
            output_path:   Destination PNG file.
            title:         Figure title.
            figsize:       ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not video_scores:
            logger.warning("No video scores to plot radar for.")
            return output_path

        # Gather axis names from the first video
        first_vid = next(iter(video_scores.values()))
        axis_names = list(first_vid.keys())
        n_axes = len(axis_names)
        if n_axes < 3:
            logger.warning("Radar chart needs ≥3 axes; skipping.")
            return output_path

        # Compute angles for each axis
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
        colours = plt.cm.tab10(np.linspace(0, 1, len(video_scores)))

        for (vid_id, scores), colour in zip(video_scores.items(), colours):
            values = [scores.get(ax, 0.0) for ax in axis_names]
            # Radar axes require non-negative values.  Scores are in [-2, 2]
            # (difference of two cosine similarities); we map this to [0, 1]
            # via (v + 2) / 4 so that the centre (0) maps to 0.5.
            values_norm = [(v + 2) / 4 for v in values]
            values_norm += values_norm[:1]
            ax.plot(angles, values_norm, "o-", linewidth=2, label=vid_id, color=colour)
            ax.fill(angles, values_norm, alpha=0.15, color=colour)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        # Tick labels show the original score values corresponding to the
        # normalised positions: 0.25 → −1, 0.5 → 0, 0.75 → +1
        ax.set_yticklabels(["−1", "0", "+1"], fontsize=8)
        ax.set_title(title, size=13, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Affective radar chart saved to %s.", output_path)
        return os.path.abspath(output_path)
