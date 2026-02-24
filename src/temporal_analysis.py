"""
temporal_analysis.py
--------------------
Analyses the *temporal* structure of a video's visual style.

The cosine-similarity matrix treats all frames as an unordered bag.  Real
marketing videos have intentional narrative rhythm — a slow atmospheric opener,
a fast-cut product reveal, a calm sign-off.  This module re-introduces time
ordering to expose:

1. **Temporal similarity curve** — smoothed per-frame similarity to its local
   neighbours, showing how visually consistent each moment is with its context.
2. **Scene-transition detection** — frames at which visual style changes sharply
   (large drop in consecutive-frame similarity).
3. **Visual pacing score** — the variance of the temporal-similarity curve.
   High variance ↔ dynamic fast-cut editing; low variance ↔ slow / contemplative.
4. **Temporal coherence score** — mean similarity between a frame and its
   immediate neighbours; higher = smoother visual narrative.
5. **Narrative-arc plot** — a Matplotlib line chart of the temporal curve with
   detected transitions marked.

These signals can be used to characterise a video's "editing DNA" and
correlate editing style with performance (CTR/ROAS) downstream.

Usage
-----
    from src.temporal_analysis import TemporalAnalyser

    ta = TemporalAnalyser(window=3)
    curve = ta.compute_temporal_curve(embeddings, index)
    transitions = ta.detect_scene_transitions(curve, threshold=0.15)
    pacing = ta.pacing_score(curve)
    coherence = ta.coherence_score(curve)
    ta.plot_narrative_arc(curve, transitions, index,
                         output_path="outputs/narrative_arc.png")
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class TemporalAnalyser:
    """
    Analyses the temporal visual-style dynamics of one or more videos.

    Args:
        window:   Half-width of the local temporal window used to compute
                  per-frame neighbourhood similarity.  A frame at position
                  ``t`` is compared to frames ``[t-window, t+window]``.
    """

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError(f"window must be ≥ 1; got {window}.")
        self.window = window

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def compute_temporal_curve(
        self,
        embeddings: np.ndarray,
        index: List[Dict],
    ) -> np.ndarray:
        """
        Compute a per-frame temporal similarity score.

        For each frame ``t`` in video ``v``, we compute the mean cosine
        similarity to the ``window`` frames immediately before and after it
        (within the same video).  Frames at the boundary use only available
        neighbours.

        The result is a dense 1-D array aligned with *embeddings*.  Frames
        belonging to different videos are analysed independently.

        Args:
            embeddings:  L2-normalised ``(N, D)`` float32 array.
            index:       Metadata list aligned with rows of *embeddings*.

        Returns:
            Float32 array ``(N,)`` of per-frame neighbourhood similarity
            scores in ``[-1, 1]``.

        Raises:
            ValueError: if embeddings is not 2-D or is empty.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {embeddings.shape}."
            )
        n = embeddings.shape[0]
        curve = np.zeros(n, dtype=np.float32)

        # Group frames by video, preserving original indices
        video_groups: Dict[str, List[int]] = {}
        for i, entry in enumerate(index):
            vid = entry.get("video_id", "unknown")
            video_groups.setdefault(vid, []).append(i)

        for vid, frame_indices in video_groups.items():
            # Sort by frame_idx / timestamp so time order is correct
            frame_indices_sorted = sorted(
                frame_indices,
                key=lambda i: index[i].get("frame_idx", index[i].get("timestamp", 0)),
            )
            m = len(frame_indices_sorted)
            for pos, global_idx in enumerate(frame_indices_sorted):
                # Window of neighbours (excluding self)
                lo = max(0, pos - self.window)
                hi = min(m - 1, pos + self.window)
                neighbour_positions = list(range(lo, pos)) + list(range(pos + 1, hi + 1))
                if not neighbour_positions:
                    curve[global_idx] = 1.0  # trivially coherent single frame
                    continue
                neighbour_indices = [frame_indices_sorted[p] for p in neighbour_positions]
                emb = embeddings[global_idx]  # (D,)
                neighbour_embs = embeddings[neighbour_indices]  # (K, D)
                sims = (neighbour_embs @ emb).astype(np.float32)
                curve[global_idx] = float(sims.mean())

        logger.info(
            "Temporal curve computed: mean=%.4f, std=%.4f, range=[%.4f, %.4f].",
            float(curve.mean()), float(curve.std()),
            float(curve.min()), float(curve.max()),
        )
        return curve

    def detect_scene_transitions(
        self,
        temporal_curve: np.ndarray,
        threshold: float = 0.15,
    ) -> List[int]:
        """
        Detect frames at which the visual style changes sharply.

        A transition is flagged at position ``t`` if the *drop* in the
        temporal-similarity curve between position ``t-1`` and ``t`` exceeds
        *threshold*.

        Args:
            temporal_curve:  Output of :meth:`compute_temporal_curve`.
            threshold:       Minimum similarity drop to count as a transition.

        Returns:
            Sorted list of 0-based frame indices that are scene-transition
            candidates.
        """
        if temporal_curve.ndim != 1 or len(temporal_curve) < 2:
            return []
        drops = temporal_curve[:-1] - temporal_curve[1:]
        transitions = [int(i + 1) for i in np.where(drops > threshold)[0]]
        logger.info(
            "Detected %d scene transitions with threshold=%.3f.",
            len(transitions), threshold,
        )
        return transitions

    def pacing_score(self, temporal_curve: np.ndarray) -> float:
        """
        Compute the **visual pacing score** of a video.

        Defined as the variance of the temporal similarity curve.

        * High variance → dynamic, fast-cut editing style.
        * Low variance  → slow, contemplative / consistent aesthetic.

        Args:
            temporal_curve:  Output of :meth:`compute_temporal_curve`.

        Returns:
            Non-negative float.
        """
        if len(temporal_curve) == 0:
            return 0.0
        return float(np.var(temporal_curve))

    def coherence_score(self, temporal_curve: np.ndarray) -> float:
        """
        Compute the **temporal coherence score** of a video.

        Defined as the mean of the temporal similarity curve.  High coherence
        means every frame is visually similar to its neighbours (smooth,
        consistent aesthetic); low coherence means frequent style jumps.

        Args:
            temporal_curve:  Output of :meth:`compute_temporal_curve`.

        Returns:
            Float in ``[-1, 1]``.
        """
        if len(temporal_curve) == 0:
            return 0.0
        return float(np.mean(temporal_curve))

    def per_video_stats(
        self,
        temporal_curve: np.ndarray,
        index: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate temporal statistics per video.

        Args:
            temporal_curve:  Output of :meth:`compute_temporal_curve`.
            index:           Metadata list aligned with *temporal_curve*.

        Returns:
            Dict ``{video_id: {coherence, pacing, n_frames, ...}}``.
        """
        video_groups: Dict[str, List[int]] = {}
        for i, entry in enumerate(index):
            vid = entry.get("video_id", "unknown")
            video_groups.setdefault(vid, []).append(i)

        stats: Dict[str, Dict[str, float]] = {}
        for vid, frame_indices in video_groups.items():
            vid_curve = temporal_curve[frame_indices]
            transitions = self.detect_scene_transitions(vid_curve)
            stats[vid] = {
                "coherence": self.coherence_score(vid_curve),
                "pacing": self.pacing_score(vid_curve),
                "n_frames": float(len(frame_indices)),
                "n_transitions": float(len(transitions)),
                "mean_sim": float(vid_curve.mean()) if len(vid_curve) else 0.0,
                "min_sim": float(vid_curve.min()) if len(vid_curve) else 0.0,
                "max_sim": float(vid_curve.max()) if len(vid_curve) else 0.0,
            }
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_temporal_stats(
        self,
        temporal_curve: np.ndarray,
        index: List[Dict],
        output_path: str,
    ) -> str:
        """
        Save per-frame temporal similarity scores to JSON.

        Format::

            [{"video_id": ..., "frame_idx": ..., "temporal_sim": 0.87, ...}]

        Args:
            temporal_curve:  ``(N,)`` array from :meth:`compute_temporal_curve`.
            index:           Metadata list aligned with *temporal_curve*.
            output_path:     Destination JSON file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = []
        for i, entry in enumerate(index):
            row = dict(entry)
            row["temporal_sim"] = round(float(temporal_curve[i]), 6) if i < len(temporal_curve) else None
            rows.append(row)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        logger.info("Temporal stats saved to %s (%d rows).", output_path, len(rows))
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_narrative_arc(
        self,
        temporal_curve: np.ndarray,
        transitions: List[int],
        index: List[Dict],
        output_path: str,
        title: str = "Narrative Arc — Temporal Visual Similarity",
        figsize: Tuple[int, int] = (14, 5),
    ) -> str:
        """
        Plot the temporal-similarity curve with scene transitions marked.

        Each video's frames are plotted as a separate coloured line segment.
        Detected scene transitions are shown as vertical dashed lines.

        Args:
            temporal_curve:  ``(N,)`` float32 array.
            transitions:     Frame indices detected as transition points.
            index:           Metadata list aligned with *temporal_curve*.
            output_path:     Destination PNG file.
            title:           Figure title.
            figsize:         ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        n = len(temporal_curve)
        x = np.arange(n)

        # Group frames by video for colour coding
        video_ids = [entry.get("video_id", "unknown") for entry in index]
        unique_vids = sorted(set(video_ids))
        cmap = plt.cm.tab10
        vid_colours = {v: cmap(i / max(len(unique_vids) - 1, 1)) for i, v in enumerate(unique_vids)}

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each video's curve segment
        for vid in unique_vids:
            vid_x = [i for i, v in enumerate(video_ids) if v == vid]
            if not vid_x:
                continue
            vid_y = temporal_curve[vid_x]
            ax.plot(vid_x, vid_y, color=vid_colours[vid], linewidth=1.5,
                    label=vid, alpha=0.9)
            ax.fill_between(vid_x, vid_y, alpha=0.08, color=vid_colours[vid])

        # Mark scene transitions
        for t_idx in transitions:
            if 0 <= t_idx < n:
                ax.axvline(t_idx, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        # Add a smoothed trend line for the overall curve
        if n >= 5:
            from numpy.lib.stride_tricks import sliding_window_view
            smooth_window = min(5, n)
            padded = np.pad(temporal_curve, (smooth_window // 2, smooth_window // 2),
                            mode="edge")
            smooth = np.convolve(padded, np.ones(smooth_window) / smooth_window,
                                 mode="valid")[:n]
            ax.plot(x, smooth, color="black", linewidth=2.0, linestyle="-",
                    alpha=0.4, label="smoothed")

        if transitions:
            ax.axvline(transitions[0], color="red", linestyle="--", linewidth=0.8,
                       alpha=0.7, label=f"transitions ({len(transitions)})")

        ax.set_xlabel("Frame index", fontsize=10)
        ax.set_ylabel("Temporal similarity to neighbours", fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title, fontsize=13, pad=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Narrative arc plot saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_pacing_comparison(
        self,
        per_video_stats: Dict[str, Dict[str, float]],
        output_path: str,
        title: str = "Visual Pacing & Coherence by Video",
        figsize: Tuple[int, int] = (10, 5),
    ) -> str:
        """
        Bar chart comparing pacing score and coherence score across videos.

        Args:
            per_video_stats:  Output of :meth:`per_video_stats`.
            output_path:      Destination PNG.
            title:            Figure title.
            figsize:          ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        vids = list(per_video_stats.keys())
        pacing = [per_video_stats[v]["pacing"] for v in vids]
        coherence = [per_video_stats[v]["coherence"] for v in vids]

        x = np.arange(len(vids))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width / 2, pacing, width, label="Pacing (variance)", color="#e07b54")
        bars2 = ax.bar(x + width / 2, coherence, width, label="Coherence (mean sim)", color="#5486e0")

        ax.set_xticks(x)
        ax.set_xticklabels(vids, fontsize=10, rotation=20, ha="right")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(title, fontsize=13, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Pacing comparison chart saved to %s.", output_path)
        return os.path.abspath(output_path)
