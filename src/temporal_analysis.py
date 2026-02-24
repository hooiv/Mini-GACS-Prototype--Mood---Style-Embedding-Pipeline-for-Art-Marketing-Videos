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

    def compute_consecutive_similarities(
        self,
        embeddings: np.ndarray,
        index: List[Dict],
    ) -> np.ndarray:
        """
        Compute the cosine similarity between each frame and its immediate
        temporal successor, within each video.

        This is the **correct signal for scene-cut detection**: a hard cut
        produces a sharp drop in ``sim(frame_t, frame_{t+1})``.  It is more
        sensitive than detecting drops in the windowed temporal curve
        (which is already a moving average — double-smoothing reduces cut
        amplitude by 30–60% in practice).

        Cross-video frame boundaries are excluded; the last frame of each
        video is assigned similarity 1.0 (no meaningful successor).

        Args:
            embeddings:  L2-normalised ``(N, D)`` float32 array.
            index:       Metadata list aligned with *embeddings*.

        Returns:
            Float32 ``(N,)`` array where entry ``t`` is
            ``cos_sim(frame_t, frame_{t+1})`` within the same video.
            Last frame of each video = 1.0.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {embeddings.shape}."
            )
        n = embeddings.shape[0]
        consec = np.ones(n, dtype=np.float32)

        video_groups: Dict[str, List[int]] = {}
        for i, entry in enumerate(index):
            vid = entry.get("video_id", "unknown")
            video_groups.setdefault(vid, []).append(i)

        for vid, frame_indices in video_groups.items():
            sorted_idx = sorted(
                frame_indices,
                key=lambda i: index[i].get("frame_idx", index[i].get("timestamp", 0)),
            )
            for pos in range(len(sorted_idx) - 1):
                cur  = embeddings[sorted_idx[pos]]
                nxt  = embeddings[sorted_idx[pos + 1]]
                consec[sorted_idx[pos]] = float(np.dot(cur, nxt))

        logger.debug(
            "Consecutive similarities: mean=%.4f, min=%.4f.",
            float(consec.mean()), float(consec.min()),
        )
        return consec

    def detect_scene_transitions(
        self,
        signal: np.ndarray,
        threshold: float = 0.15,
    ) -> List[int]:
        """
        Detect frames at which the visual style changes sharply.

        **Preferred usage**: pass the output of
        :meth:`compute_consecutive_similarities` as *signal*.  Transitions
        are then flagged at positions where the direct frame-to-frame
        similarity falls below ``1 - threshold`` (i.e. the drop exceeds
        *threshold*).

        Passing the windowed temporal curve (output of
        :meth:`compute_temporal_curve`) also works but is less precise:
        the moving-average smoothing attenuates hard-cut amplitude by
        30–60 %, causing missed detections and offset timestamps.

        Args:
            signal:     1-D float32 array — either consecutive similarities
                        ``(N,)`` from :meth:`compute_consecutive_similarities`
                        or the windowed temporal curve from
                        :meth:`compute_temporal_curve`.
            threshold:  Minimum *drop* (``signal[t-1] - signal[t]``) to count
                        as a transition.  For consecutive similarities a good
                        default is 0.15 (≈ 8° angle change).

        Returns:
            Sorted list of 0-based frame indices flagged as transition points.
        """
        if signal.ndim != 1 or len(signal) < 2:
            return []
        drops = signal[:-1] - signal[1:]
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

        Note: prefer :meth:`pacing_rate_per_second` when timestamps are
        available; it is more interpretable (cuts/minute) and does not depend
        on the scale of the similarity values.

        Args:
            temporal_curve:  Output of :meth:`compute_temporal_curve`.

        Returns:
            Non-negative float.
        """
        if len(temporal_curve) == 0:
            return 0.0
        return float(np.var(temporal_curve))

    def pacing_rate_per_second(
        self,
        transitions: List[int],
        index: List[Dict],
        video_id: Optional[str] = None,
    ) -> float:
        """
        Compute the scene-transition rate in **transitions per second**.

        This is a more interpretable pacing metric than variance of the
        similarity curve because it is independent of embedding scale and
        directly comparable across videos of different lengths.

        Args:
            transitions:  Transition frame indices from
                          :meth:`detect_scene_transitions`.
            index:        Metadata list aligned with the embeddings used to
                          produce *transitions*.
            video_id:     If given, only frames belonging to this video are
                          used to compute the duration.  Otherwise all frames
                          are used.

        Returns:
            Float in ``[0, ∞)``.  Returns 0.0 if no timestamps are available
            or the video has zero duration.
        """
        if not transitions:
            return 0.0

        # Extract timestamps for the relevant frames
        timestamps = []
        for entry in index:
            if video_id is not None and entry.get("video_id") != video_id:
                continue
            ts = entry.get("timestamp")
            if ts is not None:
                timestamps.append(float(ts))

        if len(timestamps) < 2:
            return 0.0

        duration = max(timestamps) - min(timestamps)
        if duration <= 0:
            return 0.0

        # Only count transitions that fall within this video's frame indices
        if video_id is not None:
            vid_indices = {
                i for i, e in enumerate(index)
                if e.get("video_id") == video_id
            }
            n_transitions = sum(1 for t in transitions if t in vid_indices)
        else:
            n_transitions = len(transitions)

        return n_transitions / duration

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
