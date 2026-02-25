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

# ---------------------------------------------------------------------------
# Multi-prompt axis definitions (5 phrasings per pole)
#
# Using a single prompt per pole risks conflating CLIP's response to specific
# words (e.g. "golden") with the underlying affective concept.  Ensembling
# multiple semantically-equivalent prompts and averaging their embeddings
# (in embedding space, before computing similarity) cancels per-prompt noise
# and yields a more stable axis direction — the same technique used in the
# original CLIP zero-shot classification paper (Radford et al. 2021, §3.1).
# ---------------------------------------------------------------------------
MULTI_PROMPT_AXES: Dict[str, Tuple[List[str], List[str]]] = {
    "energy": (
        [
            "an energetic, dynamic, fast-paced scene with motion and action",
            "a vibrant, kinetic, high-intensity visual",
            "explosive movement and dynamic energy in the frame",
            "fast cuts, motion blur, action and excitement",
            "a high-energy advertisement with rapid visual transitions",
        ],
        [
            "a calm, still, peaceful and quiet scene",
            "a serene, slow, meditative visual with no movement",
            "motionless, static, tranquil imagery",
            "a still life photograph with no energy or motion",
            "a slow-paced, contemplative, restful scene",
        ],
    ),
    "warmth": (
        [
            "a warm, golden, sun-lit scene bathed in warm orange tones",
            "a cozy, amber-lit interior with warm, inviting colours",
            "sunset hues, warm reds and yellows, a welcoming atmosphere",
            "golden-hour lighting with a warm, comfortable mood",
            "a scene with warm colour grading, orange and yellow tones",
        ],
        [
            "a cold, blue-toned, icy scene with cool lighting",
            "a clinical, sterile environment with cool white light",
            "frozen, wintry, pale blue and grey colour palette",
            "a cold night scene with blue-tinted shadows",
            "a scene with cold colour grading, blue and teal tones",
        ],
    ),
    "complexity": (
        [
            "a visually complex, highly detailed, densely textured scene",
            "a busy, crowded frame packed with visual information",
            "intricate patterns, many overlapping elements, visual noise",
            "a chaotic, layered composition with numerous objects",
            "highly detailed fine art with complex visual structure",
        ],
        [
            "a minimalist, clean, uncluttered scene with empty space",
            "simple geometric shapes on a plain background",
            "a spartan composition with a single subject and no distractions",
            "negative space, clean lines, minimal visual elements",
            "a plain, empty, featureless background",
        ],
    ),
    "luxury": (
        [
            "an opulent, high-end, luxurious aesthetic with gold and marble",
            "a premium, exclusive product in an elegant setting",
            "haute couture fashion, fine jewellery, luxury brand imagery",
            "a sophisticated, polished, aspirational visual",
            "rich materials, soft lighting, and a refined, tasteful aesthetic",
        ],
        [
            "a raw, gritty, low-budget aesthetic with rough textures",
            "a DIY, home-made, unpolished visual",
            "cheap materials, harsh lighting, and an unrefined look",
            "a rough, industrial, low-production-value scene",
            "a basic, utilitarian environment with no aesthetic polish",
        ],
    ),
    "joy": (
        [
            "a joyful, happy, uplifting scene full of smiles and laughter",
            "people celebrating, radiating happiness and positive energy",
            "bright, cheerful colours and a feel-good, optimistic mood",
            "a sunny, carefree, delightful visual experience",
            "warmth and happiness, children playing, pure joy",
        ],
        [
            "a dark, melancholic, sad and sorrowful scene",
            "grief, loneliness, and emotional pain captured visually",
            "a gloomy, overcast, desaturated scene with a heavy mood",
            "despair, isolation, and sadness in a bleak environment",
            "a sombre, muted, emotionally heavy visual",
        ],
    ),
    "tension": (
        [
            "a tense, dramatic, suspenseful scene with high stakes",
            "an ominous, foreboding visual with a sense of danger",
            "dark shadows, dramatic contrast, and a threatening atmosphere",
            "a thriller or horror visual with palpable tension",
            "conflict, confrontation, and nervous anticipation",
        ],
        [
            "a relaxed, serene, tension-free and peaceful scene",
            "a gentle, safe, comfortable environment with no threat",
            "an idyllic, stress-free scene of calm and contentment",
            "soft lighting, gentle colours, and a totally relaxed mood",
            "a harmonious, conflict-free scene full of ease and calm",
        ],
    ),
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

        # Pre-compute and cache text embeddings for both scoring modes.
        # This eliminates redundant CLIP forward passes on every scoring call:
        #   score_frames()          → 2 prompts/axis × N_axes = 12 passes → 0
        #   score_frames_ensemble() → 10 prompts/axis × N_axes = 60 → 0
        # Dict layout:
        #   _single_cache[axis_name] = (pos_emb (D,), neg_emb (D,))
        #   _ensemble_cache[axis_name] = (pos_anchor (D,), neg_anchor (D,),
        #                                  pos_embs (K,D), neg_embs (K,D))
        self._single_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._ensemble_cache: Dict[
            str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._prewarm_single_prompt_cache()
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

    def _prewarm_single_prompt_cache(self) -> None:
        """
        Eagerly encode all single-prompt axis pairs and populate
        ``_single_cache``.  Called once at the end of ``__init__``.
        """
        for axis_name, (pos_prompt, neg_prompt) in self.axes.items():
            embs = self.encode_text([pos_prompt, neg_prompt])
            self._single_cache[axis_name] = (embs[0], embs[1])
        logger.debug(
            "Single-prompt cache pre-warmed for %d axes.", len(self.axes)
        )

    def invalidate_cache(self) -> None:
        """
        Clear both prompt-embedding caches.

        Call this after modifying ``self.axes`` at runtime to force
        re-encoding on the next :meth:`score_frames` /
        :meth:`score_frames_ensemble` call.
        """
        self._single_cache.clear()
        self._ensemble_cache.clear()
        logger.debug("Affective scoring prompt caches invalidated.")

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
            # Use cached embeddings if available; populate cache on miss.
            if axis_name not in self._single_cache:
                text_embs = self.encode_text([pos_prompt, neg_prompt])
                self._single_cache[axis_name] = (text_embs[0], text_embs[1])
            pos_emb, neg_emb = self._single_cache[axis_name]

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

    def score_frames_ensemble(
        self,
        frame_embeddings: np.ndarray,
        multi_prompt_axes: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
        index: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Multi-prompt ensembled affective scoring with confidence estimates.

        Single-prompt probing conflates CLIP's response to specific word
        choices (e.g. "golden") with the underlying affective concept.
        This method encodes *K* positive prompts and *K* negative prompts per
        axis, averages the resulting text embeddings (in embedding space,
        before computing similarity), and scores each frame against the
        averaged anchor vectors.

        In addition it computes a **confidence score** per axis per frame:
        the standard deviation of scores across the *K* individual prompt
        pairs.  Low std → the axis direction is stable regardless of phrasing
        (high confidence); high std → prompt wording strongly influences the
        score (treat the result with scepticism).

        This directly mirrors the ensemble technique from the original CLIP
        zero-shot classification paper (Radford et al. 2021, §3.1).

        Args:
            frame_embeddings:   Float32 ``(N, D)`` L2-normalised image embeddings.
            multi_prompt_axes:  Dict ``{axis: ([pos_prompts], [neg_prompts])}``.
                                Defaults to :data:`MULTI_PROMPT_AXES`.
            index:              Ignored (kept for API symmetry).

        Returns:
            Tuple ``(mean_scores, confidence_scores)`` where both are dicts
            mapping axis name → float32 ``(N,)`` array.

            - *mean_scores*: ensemble-averaged axis score for each frame.
              Range ``[-2, 2]``.
            - *confidence_scores*: per-frame std of scores across prompt pairs.
              Lower = more confident axis direction.

        Raises:
            ValueError: if ``frame_embeddings`` is not 2-D or is empty.
        """
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {frame_embeddings.shape}."
            )

        axes_def = multi_prompt_axes or MULTI_PROMPT_AXES
        n = frame_embeddings.shape[0]
        # Cache is only safe to use when the default MULTI_PROMPT_AXES are in
        # play; a custom dict may have different prompts under the same names.
        use_cache = multi_prompt_axes is None
        mean_scores: Dict[str, np.ndarray] = {}
        confidence_scores: Dict[str, np.ndarray] = {}
        k = 0  # initialised in the loop; declared here for the log message

        for axis_name, (pos_prompts, neg_prompts) in axes_def.items():
            # Serve from cache when possible, populate on miss.
            if use_cache and axis_name in self._ensemble_cache:
                pos_anchor, neg_anchor, pos_embs, neg_embs = (
                    self._ensemble_cache[axis_name]
                )
            else:
                all_prompts = pos_prompts + neg_prompts
                all_embs = self.encode_text(all_prompts)  # (2K, D)
                k = len(pos_prompts)
                pos_embs = all_embs[:k]   # (K, D) — already L2-normalised
                neg_embs = all_embs[k:]   # (K, D)

                # Mean-pool in embedding space → average axis anchor vector,
                # then re-normalise so the anchor remains a unit vector.
                pos_anchor = pos_embs.mean(axis=0)
                pos_anchor /= max(np.linalg.norm(pos_anchor), 1e-8)
                neg_anchor = neg_embs.mean(axis=0)
                neg_anchor /= max(np.linalg.norm(neg_anchor), 1e-8)

                if use_cache:
                    self._ensemble_cache[axis_name] = (
                        pos_anchor, neg_anchor, pos_embs, neg_embs
                    )

            k = pos_embs.shape[0]

            # Ensemble mean score
            mean_axis_score = np.clip(
                frame_embeddings @ pos_anchor - frame_embeddings @ neg_anchor,
                -2.0, 2.0,
            ).astype(np.float32)

            # Per-prompt-pair scores → std as confidence proxy
            pair_scores = np.zeros((k, n), dtype=np.float32)
            for pi in range(k):
                p = pos_embs[pi]
                q = neg_embs[pi]
                pair_scores[pi] = np.clip(
                    frame_embeddings @ p - frame_embeddings @ q, -2.0, 2.0
                )
            axis_confidence = pair_scores.std(axis=0).astype(np.float32)

            mean_scores[axis_name] = mean_axis_score
            confidence_scores[axis_name] = axis_confidence

            logger.debug(
                "Ensemble axis '%s': mean=%.4f, mean_confidence=%.4f.",
                axis_name,
                float(mean_axis_score.mean()),
                float(axis_confidence.mean()),
            )

        logger.info(
            "Ensemble affective scores computed for %d frames, %d axes, "
            "using %d prompts per pole.",
            n, len(axes_def), k,
        )
        return mean_scores, confidence_scores

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

    def score_frames_gap_corrected(
        self,
        frame_embeddings: np.ndarray,
        index: Optional[List[Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute affective scores with CLIP modality-gap correction applied.

        Calls :class:`~src.modality_alignment.ModalityAligner` to remove the
        systematic bias caused by image and text embeddings occupying different
        cones of the unit sphere (Liang et al. NeurIPS 2022).  See
        ``src/modality_alignment.py`` for a full explanation of the problem
        and the correction algorithm.

        The correction shifts both modalities by half the gap vector before
        computing cosine similarities.  After correction, axis scores are more
        sensitive to genuine semantic content and less influenced by the
        structural offset between modalities.

        Args:
            frame_embeddings:  L2-normalised image embeddings ``(N, D)``.
            index:             Optional metadata list (passed through unused).

        Returns:
            Same format as :meth:`score_frames`: ``Dict[axis_name → (N,)
            float32]`` with values clipped to ``[-2, 2]``.

        Raises:
            ValueError: if ``frame_embeddings`` is not 2-D or is empty.
        """
        from src.modality_alignment import ModalityAligner

        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {frame_embeddings.shape}."
            )

        # Collect all text anchors for fitting the aligner
        all_text_embs = []
        for axis_name, (pos_prompt, neg_prompt) in self.axes.items():
            pair_embs = self.encode_text([pos_prompt, neg_prompt])
            all_text_embs.append(pair_embs)
        text_matrix = np.vstack(all_text_embs)  # (2 * n_axes, D)

        aligner = ModalityAligner()
        aligner.fit(frame_embeddings, text_matrix)

        corrected_frames = aligner.correct_image(frame_embeddings)

        axis_scores: Dict[str, np.ndarray] = {}
        for axis_name, (pos_prompt, neg_prompt) in self.axes.items():
            text_embs = self.encode_text([pos_prompt, neg_prompt])
            corr_pos  = aligner.correct_text(text_embs[[0]])
            corr_neg  = aligner.correct_text(text_embs[[1]])

            pos_sims = (corrected_frames @ corr_pos[0]).astype(np.float32)
            neg_sims = (corrected_frames @ corr_neg[0]).astype(np.float32)
            scores   = np.clip(pos_sims - neg_sims, -2.0, 2.0)
            axis_scores[axis_name] = scores

        gap_mag = aligner.gap_magnitude
        logger.info(
            "Gap-corrected scoring complete "
            "(gap_magnitude=%.4f, N=%d, axes=%d).",
            gap_mag, frame_embeddings.shape[0], len(self.axes),
        )
        return axis_scores

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

        # Use robust percentile-based colour range.
        # Hardcoding vmin/vmax=-0.5/0.5 clips 75% of the [-2, 2] dynamic
        # range of ensemble scores, washing all frames to the same mid-colour.
        all_vals = matrix.ravel()
        v_abs = max(abs(float(np.percentile(all_vals, 5))),
                    abs(float(np.percentile(all_vals, 95))),
                    1e-4)
        vmin, vmax = -v_abs, v_abs

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
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
