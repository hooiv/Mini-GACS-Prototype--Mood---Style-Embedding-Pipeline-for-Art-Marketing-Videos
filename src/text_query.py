"""
text_query.py
-------------
Text-guided creative retrieval using CLIP's joint image-text embedding space.

Problem
~~~~~~~
The entire pipeline so far treats retrieval as "find visually similar frames
to frame X" — an *image-to-image* query.  In real creative intelligence
workflows the primary use case is the inverse:

    "Show me frames that match this brand brief."
    "Find creatives with a luxurious, warm, minimalist mood."
    "Which video best aligns with our Q4 holiday campaign tone?"

CLIP's killer feature is that text and images are embedded in the *same*
cosine-similarity space.  A CLIP text embedding for "warm, golden, cinematic"
sits close to image embeddings of frames that evoke exactly that mood —
without any labelled training data, and without any visual similarity to a
specific reference image.

This module implements a ``TextQueryRetriever`` that:

1. Encodes a free-text query (or brand brief) with the CLIP text encoder.
2. Computes cosine similarity between the text embedding and all stored frame
   embeddings — yielding a per-frame "brief alignment score".
3. Returns a ranked list of ``TextQueryResult`` objects with similarity scores.
4. Aggregates frame-level scores to video level (``rank_videos_by_brief``),
   using *mean-of-top-3* aggregation — robust to a single exceptional frame
   inflating a video's score.
5. Compares multiple briefs against all videos in one call
   (``multi_brief_comparison``), producing a briefs × videos matrix suitable
   for a heatmap.

Architecture
~~~~~~~~~~~~
::

    text query (free text)
          │
          ▼
    CLIP text encoder  ──►  L2-normalised query embedding (D,)
          │
          ▼
    dot product with frame embeddings (N, D)  ──►  alignment scores (N,)
          │
          ▼
    argsort descending  ──►  top-k TextQueryResult objects
          │
          ▼
    rank_videos_by_brief()  ──►  per-video mean-top-3 score
    multi_brief_comparison() ──►  Dict[brief_name, Dict[video_id, score]]

Usage
-----
    from src.text_query import TextQueryRetriever

    # Initialise with a real CLIP model (lazy-loads on first call if None)
    retriever = TextQueryRetriever(model=clip_model, processor=clip_processor)

    # Single text query → top-10 matching frames
    results = retriever.query(
        "warm, golden, cinematic luxury aesthetic",
        embeddings, index, top_k=10,
    )
    retriever.plot_query_results(results, "outputs/text_query.png")
    retriever.save_results(results, "outputs/text_query.json")

    # Rank entire videos by a brand brief
    video_ranking = retriever.rank_videos_by_brief(
        "minimalist, modern, high-end brand identity",
        embeddings, index,
    )

    # Compare multiple briefs
    comparison = retriever.multi_brief_comparison(
        {"holiday": "warm, festive, joyful", "luxury": "dark, opulent, exclusive"},
        embeddings, index,
    )
    retriever.plot_brief_comparison_heatmap(comparison, "outputs/brief_heatmap.png")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Mean-top-k aggregation: average the top-K frame scores per video.
# Using the top-3 (not the mean of all frames) is more robust: if 90 % of a
# video is slow static shots and 3 frames are perfectly on-brief, the
# mean-all aggregation would bury those matches.  Mean-top-3 surfaces them.
_MEAN_TOP_K: int = 3


@dataclass
class TextQueryResult:
    """A single frame result from a text-query search."""

    rank: int           # 1-based rank (1 = closest match to the text query)
    frame_idx: int      # Row index in the embeddings / index arrays
    video_id: str       # Source video identifier
    timestamp: float    # Frame timestamp in seconds
    similarity: float   # CLIP cosine similarity ∈ [-1, 1]; higher = better match
    metadata: Dict      # Full original frame metadata dict


class TextQueryRetriever:
    """
    Text-guided frame retrieval using CLIP's joint image-text embedding space.

    Args:
        model:      A ``CLIPModel`` instance (HuggingFace).  If ``None``, the
                    default checkpoint is loaded lazily on first use.
        processor:  A ``CLIPProcessor`` instance.  Must be provided when
                    *model* is not None.
        model_name: HuggingFace checkpoint used if ``model`` is None.
        device:     Torch device string.  Auto-detected when None.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        self._model = model
        self._processor = processor
        self._model_name = model_name
        self._device = device
        self._embedding_model = None  # lazy EmbeddingModel wrapper

    # ------------------------------------------------------------------
    # Internal: lazy CLIP initialisation
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Return a live EmbeddingModel, lazy-loading if necessary."""
        if self._model is not None and self._processor is not None:
            # Thin wrapper that exposes only encode_text using the provided
            # model/processor — avoids creating a second model instance.
            return _InlineTextEncoder(self._model, self._processor, self._device)
        if self._embedding_model is None:
            from src.embeddings import EmbeddingModel  # local import avoids circular
            self._embedding_model = EmbeddingModel(
                model_name=self._model_name,
                device=self._device,
            )
        return self._embedding_model

    # ------------------------------------------------------------------
    # Core text encoding
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a text string to an L2-normalised CLIP text embedding.

        Args:
            text:  Free-text query (e.g. "warm golden luxury").

        Returns:
            Float32 array ``(D,)`` — same dimensionality as frame embeddings.
        """
        enc = self._get_embedding_model()
        vec = enc.encode_text([text])
        return vec[0]

    # ------------------------------------------------------------------
    # Frame-level retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        embeddings: np.ndarray,
        index: List[Dict],
        top_k: int = 10,
    ) -> List[TextQueryResult]:
        """
        Retrieve the top-*k* frames that best match a text query.

        The query text is encoded by the CLIP text encoder.  Its L2-normalised
        embedding is dot-producted with all frame embeddings (also L2-normalised
        by :class:`src.embeddings.EmbeddingModel`), yielding per-frame cosine
        similarity scores in ``[-1, 1]``.  The top-*k* frames are returned in
        descending order.

        Args:
            text:        Free-text query or brand brief.
            embeddings:  L2-normalised float32 ``(N, D)`` frame embeddings.
            index:       Metadata list aligned with *embeddings* rows.
            top_k:       Maximum number of results to return.

        Returns:
            List of at most *top_k* :class:`TextQueryResult` objects,
            sorted by descending similarity.

        Raises:
            ValueError: if *embeddings* is empty or not 2-D.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty embeddings; got shape {embeddings.shape}."
            )

        query_emb = self.encode_query(text)  # (D,)
        sims = (embeddings @ query_emb).astype(np.float32)
        sims = np.clip(sims, -1.0, 1.0)

        top_k = min(top_k, len(sims))
        top_idx = np.argsort(sims)[::-1][:top_k]

        results: List[TextQueryResult] = []
        for rank_pos, frame_idx in enumerate(top_idx):
            if frame_idx >= len(index):
                continue
            meta = index[int(frame_idx)]
            results.append(
                TextQueryResult(
                    rank=rank_pos + 1,
                    frame_idx=int(frame_idx),
                    video_id=str(meta.get("video_id", "unknown")),
                    timestamp=float(meta.get("timestamp", 0.0)),
                    similarity=float(sims[frame_idx]),
                    metadata=dict(meta),
                )
            )

        logger.info(
            "Text query '%s': top-%d matches, best=%.4f, worst=%.4f.",
            text[:60], len(results),
            results[0].similarity if results else float("nan"),
            results[-1].similarity if results else float("nan"),
        )
        return results

    # ------------------------------------------------------------------
    # Video-level brief alignment
    # ------------------------------------------------------------------

    def rank_videos_by_brief(
        self,
        text: str,
        embeddings: np.ndarray,
        index: List[Dict],
        aggregation: str = "mean_top3",
    ) -> List[Dict[str, Any]]:
        """
        Score and rank entire videos by alignment to a text brief.

        Frame-level similarity scores are aggregated to video level using one
        of three strategies:

        - ``"mean"``: average similarity across all frames in the video.
          Sensitive to the ratio of on-brief vs off-brief frames.
        - ``"mean_top3"``: average the top-3 frame similarities.  More robust
          than *mean* when most frames are off-brief (e.g. a 5-minute video
          where only 3 frames show the target product).
        - ``"max"``: highest single-frame similarity.  Asks "does the video
          contain at least one perfect match?"

        The default ``"mean_top3"`` is the recommended production choice: it
        rewards genuinely on-brief content without being gamed by a single
        accidentally matching frame.

        Args:
            text:         Text brief for the query.
            embeddings:   L2-normalised ``(N, D)`` frame embeddings.
            index:        Metadata list aligned with *embeddings* rows.
            aggregation:  ``"mean"``, ``"mean_top3"``, or ``"max"``.

        Returns:
            List of dicts ``{"video_id", "score", "n_frames"}`` sorted by
            descending score.

        Raises:
            ValueError: if *aggregation* is unknown or *embeddings* is empty.
        """
        if aggregation not in ("mean", "mean_top3", "max"):
            raise ValueError(
                f"Unknown aggregation '{aggregation}'; "
                "use 'mean', 'mean_top3', or 'max'."
            )
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty embeddings; got shape {embeddings.shape}."
            )

        query_emb = self.encode_query(text)
        sims = np.clip((embeddings @ query_emb).astype(np.float32), -1.0, 1.0)

        # Group frame indices by video_id
        video_frames: Dict[str, List[int]] = {}
        for i, meta in enumerate(index):
            vid = str(meta.get("video_id", "unknown"))
            video_frames.setdefault(vid, []).append(i)

        ranking: List[Dict[str, Any]] = []
        for vid, frame_indices in video_frames.items():
            vid_sims = sims[frame_indices]
            if aggregation == "mean":
                score = float(vid_sims.mean())
            elif aggregation == "mean_top3":
                k = min(_MEAN_TOP_K, len(vid_sims))
                score = float(np.sort(vid_sims)[::-1][:k].mean())
            else:  # max
                score = float(vid_sims.max())
            ranking.append(
                {"video_id": vid, "score": score, "n_frames": len(frame_indices)}
            )

        ranking.sort(key=lambda d: d["score"], reverse=True)
        logger.info(
            "rank_videos_by_brief('%s', agg=%s): %d videos ranked.",
            text[:60], aggregation, len(ranking),
        )
        return ranking

    # ------------------------------------------------------------------
    # Multi-brief comparison
    # ------------------------------------------------------------------

    def multi_brief_comparison(
        self,
        briefs: Dict[str, str],
        embeddings: np.ndarray,
        index: List[Dict],
        aggregation: str = "mean_top3",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple text briefs against all videos in a single call.

        Useful for a creative intelligence dashboard: each brief (column) is
        evaluated against each video (row), and the result is a matrix of
        alignment scores.

        Args:
            briefs:       Dict ``{brief_name: brief_text}``.
            embeddings:   L2-normalised ``(N, D)`` frame embeddings.
            index:        Metadata list aligned with *embeddings* rows.
            aggregation:  Aggregation strategy (see :meth:`rank_videos_by_brief`).

        Returns:
            Nested dict ``{brief_name: {video_id: score}}``.

        Raises:
            ValueError: if *briefs* is empty.
        """
        if not briefs:
            raise ValueError("briefs must be a non-empty dict.")

        result: Dict[str, Dict[str, float]] = {}
        for brief_name, brief_text in briefs.items():
            video_ranking = self.rank_videos_by_brief(
                brief_text, embeddings, index, aggregation=aggregation
            )
            result[brief_name] = {row["video_id"]: row["score"] for row in video_ranking}

        logger.info(
            "multi_brief_comparison: %d briefs × %d videos.",
            len(briefs),
            len(next(iter(result.values()))) if result else 0,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: List[TextQueryResult],
        output_path: str,
    ) -> str:
        """
        Save text-query results to JSON.

        Args:
            results:      Output of :meth:`query`.
            output_path:  Destination file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = [asdict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        logger.info("Text query results saved to %s (%d entries).", output_path, len(rows))
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_query_results(
        self,
        results: List[TextQueryResult],
        output_path: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 5),
    ) -> str:
        """
        Horizontal bar chart of similarity scores for the top-k query results.

        Bars are coloured by source video.  The query text (truncated to
        60 characters) is included in the figure title.

        Args:
            results:      Output of :meth:`query`.
            output_path:  Destination PNG file.
            title:        Override figure title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not results:
            logger.warning("Empty results — skipping plot.")
            return output_path

        labels = [
            f"#{r.rank}  {r.video_id} @ {r.timestamp:.1f}s" for r in results
        ]
        scores = [r.similarity for r in results]

        unique_vids = sorted(set(r.video_id for r in results))
        cmap = plt.cm.tab10
        vid_colour = {
            v: cmap(i / max(len(unique_vids) - 1, 1))
            for i, v in enumerate(unique_vids)
        }
        colours = [vid_colour[r.video_id] for r in results]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = list(range(len(results)))
        ax.barh(y_pos, scores, color=colours, alpha=0.82, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("CLIP cosine similarity to text query", fontsize=10)
        ax.set_title(title or "Text Query Results", fontsize=12, pad=10)
        ax.invert_yaxis()
        ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor=vid_colour[v], label=v, alpha=0.82)
            for v in unique_vids
        ]
        ax.legend(handles=legend_els, fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Text query chart saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_brief_comparison_heatmap(
        self,
        comparison: Dict[str, Dict[str, float]],
        output_path: str,
        title: str = "Brand Brief × Video Alignment",
        figsize: Tuple[int, int] = (9, 6),
    ) -> str:
        """
        Heatmap of briefs (rows) × videos (columns) alignment scores.

        Each cell shows the mean-top-3 alignment score.  This gives a
        quick overview of which video best matches each campaign brief.

        Args:
            comparison:   Output of :meth:`multi_brief_comparison`.
            output_path:  Destination PNG.
            title:        Figure title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not comparison:
            logger.warning("Empty comparison — skipping heatmap.")
            return output_path

        brief_names = sorted(comparison.keys())
        video_ids = sorted({vid for brief_scores in comparison.values() for vid in brief_scores})

        matrix = np.array(
            [
                [comparison[b].get(v, float("nan")) for v in video_ids]
                for b in brief_names
            ],
            dtype=np.float32,
        )

        fig, ax = plt.subplots(figsize=figsize)
        # Use percentile-based colour range so narrow spreads are visible
        all_vals = matrix[~np.isnan(matrix)]
        vmin = float(np.percentile(all_vals, 5)) if all_vals.size else 0.0
        vmax = float(np.percentile(all_vals, 95)) if all_vals.size else 1.0
        if vmax - vmin < 1e-4:
            vmin, vmax = float(np.min(all_vals)) - 0.01, float(np.max(all_vals)) + 0.01

        im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(video_ids)))
        ax.set_xticklabels(video_ids, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(brief_names)))
        ax.set_yticklabels(brief_names, fontsize=9)
        ax.set_title(title, fontsize=12, pad=10)

        # Annotate cells
        for i in range(len(brief_names)):
            for j in range(len(video_ids)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if val < (vmin + vmax) / 2 else "white",
                    )

        plt.colorbar(im, ax=ax, label="Alignment score (CLIP cosine sim)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Brief comparison heatmap saved to %s.", output_path)
        return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Internal helper: thin text encoder wrapping an externally provided model
# ---------------------------------------------------------------------------

class _InlineTextEncoder:
    """
    Thin wrapper that exposes only ``encode_text`` using an externally provided
    CLIPModel + CLIPProcessor.  Avoids creating a second model instance when
    the caller already has a loaded model (typical in tests and main.py).
    """

    def __init__(self, model, processor, device: Optional[str] = None) -> None:
        import torch
        self._model = model
        self._processor = processor
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        import torch

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        result = feats.cpu().numpy().astype(np.float32)

        if np.isnan(result).any():
            raise ValueError("Text embeddings contain NaN values.")
        return result
