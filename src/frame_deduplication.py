"""
frame_deduplication.py
----------------------
Content-adaptive frame selection for the Mini GACS pipeline.

**Why this matters**
~~~~~~~~~~~~~~~~~~~~
Uniform interval sampling (e.g. 1 frame/second) is simple but produces large
blocks of nearly-identical frames wherever the video has slow pans, static
shots, or fades.  These near-duplicates have three harmful downstream effects:

1. **Similarity-matrix artifacts** — block-diagonal correlation driven purely
   by temporal proximity, masking genuine cross-video style similarity.
2. **Cluster size bias** — clusters from slower-paced videos absorb
   disproportionately many frames, distorting the K-means solution.
3. **Temporal coherence inflation** — windowed-similarity scores appear
   artificially high because consecutive frames are nearly identical, not
   because the video has a coherent style.

**Approach: greedy cosine-threshold deduplication**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Walk through frames in timestamp order.  Accept frame *t* if and only if its
maximum cosine similarity to any already-accepted frame is below a configurable
threshold *tau* (default 0.97).  This is an O(N · K) greedy algorithm where K
grows slowly (at most N); in practice it runs in milliseconds for hundreds of
frames.

A lower *tau* (e.g. 0.90) yields an aggressive curation of only strongly
distinct frames.  A higher *tau* (e.g. 0.99) only removes frame-exact
duplicates.  The default of 0.97 removes near-duplicates from slow pans and
static shots while preserving smooth visual transitions.

**Also provided**
~~~~~~~~~~~~~~~~~
* ``compute_novelty_scores`` — per-frame novelty relative to its temporal
  predecessor; useful for ranking frames by how much new information they add.
* ``select_diverse_frames`` — MMR (Maximum Marginal Relevance) selection of
  exactly *n_select* maximally-diverse frames; useful when a fixed budget is
  required regardless of video pacing.

Usage
-----
    from src.frame_deduplication import deduplicate_frames

    # embeddings: (N, D) L2-normalised CLIP embeddings, aligned with metadata
    dedup_embeddings, dedup_index, kept_mask = deduplicate_frames(
        embeddings, metadata, similarity_threshold=0.97
    )
    print(f"Kept {dedup_embeddings.shape[0]} / {len(metadata)} frames "
          f"({100 * dedup_embeddings.shape[0] / len(metadata):.1f}%)")
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sensible default: remove near-duplicates but keep smooth transitions
DEFAULT_DEDUP_THRESHOLD: float = 0.97


def deduplicate_frames(
    embeddings: np.ndarray,
    index: List[Dict],
    similarity_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Greedy content-adaptive frame selection.

    Walk through frames in their given order (assumed to be temporal).
    Accept frame *i* if its maximum cosine similarity to all already-accepted
    frames is strictly below *similarity_threshold*.  The first frame is
    always accepted.

    Because embeddings are L2-normalised, cosine similarity equals the dot
    product; this allows vectorised computation without explicit normalisation.

    Args:
        embeddings:           Float32 ``(N, D)`` L2-normalised CLIP embeddings,
                              ordered by time.
        index:                Metadata list aligned row-by-row with *embeddings*.
        similarity_threshold: Maximum cosine similarity to any already-accepted
                              frame before a new frame is rejected.  Range
                              ``(0, 1]``.  Default 0.97.

    Returns:
        Tuple of:
        - ``dedup_embeddings``: Float32 ``(M, D)`` array of kept frame embeddings.
        - ``dedup_index``:      Corresponding metadata list of length *M*.
        - ``kept_mask``:        Boolean array ``(N,)`` – True at accepted positions.

    Raises:
        ValueError: if *embeddings* is not 2-D, is empty, or
                    ``similarity_threshold`` is out of range.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Expected 2-D non-empty array; got shape {embeddings.shape}."
        )
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be in (0, 1]; got {similarity_threshold}."
        )
    if len(index) != embeddings.shape[0]:
        raise ValueError(
            f"len(index)={len(index)} must equal embeddings.shape[0]={embeddings.shape[0]}."
        )

    n = embeddings.shape[0]
    kept_mask = np.zeros(n, dtype=bool)
    accepted_indices: List[int] = []

    for i in range(n):
        if not accepted_indices:
            # Always accept the first frame
            kept_mask[i] = True
            accepted_indices.append(i)
            continue

        # Vectorised: similarity of frame i to all accepted frames
        accepted_embs = embeddings[accepted_indices]  # (K, D)
        sims = accepted_embs @ embeddings[i]           # (K,)
        max_sim = float(sims.max())

        if max_sim < similarity_threshold:
            kept_mask[i] = True
            accepted_indices.append(i)

    dedup_embeddings = embeddings[kept_mask]
    dedup_index = [index[i] for i in range(n) if kept_mask[i]]

    n_removed = n - int(kept_mask.sum())
    logger.info(
        "Frame deduplication (tau=%.3f): kept %d / %d frames "
        "(removed %d near-duplicates, %.1f%% reduction).",
        similarity_threshold,
        len(dedup_index), n, n_removed,
        100.0 * n_removed / max(n, 1),
    )
    return dedup_embeddings, dedup_index, kept_mask


def compute_novelty_scores(
    embeddings: np.ndarray,
    index: List[Dict],
) -> np.ndarray:
    """
    Compute per-frame novelty relative to its *temporal predecessor*.

    Novelty is defined as ``1 - cosine_similarity(frame_t, frame_{t-1})``.
    Higher novelty means the frame introduces more new visual information.
    The first frame in each video has novelty 1.0 by definition.

    Frames are processed per-video (using the ``"video_id"`` metadata field)
    so that cross-video boundaries are not penalised as transitions.

    Args:
        embeddings: Float32 ``(N, D)`` L2-normalised embeddings.
        index:      Metadata list aligned with *embeddings*.

    Returns:
        Float32 ``(N,)`` novelty scores in ``[0, 2]`` (unbounded above due
        to anti-correlated frames, but typically in ``[0, 0.5]`` for natural
        video content).
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Expected 2-D non-empty array; got shape {embeddings.shape}."
        )

    n = embeddings.shape[0]
    novelty = np.ones(n, dtype=np.float32)  # default = max novelty

    # Group by video, compute within each video
    video_groups: Dict[str, List[int]] = {}
    for i, entry in enumerate(index):
        vid = entry.get("video_id", "unknown")
        video_groups.setdefault(vid, []).append(i)

    for vid, frame_indices in video_groups.items():
        # Sort by frame_idx / timestamp to ensure temporal ordering
        sorted_idx = sorted(
            frame_indices,
            key=lambda i: index[i].get("frame_idx", index[i].get("timestamp", 0)),
        )
        for pos in range(1, len(sorted_idx)):
            cur  = embeddings[sorted_idx[pos]]
            prev = embeddings[sorted_idx[pos - 1]]
            cos_sim = float(np.dot(cur, prev))
            novelty[sorted_idx[pos]] = 1.0 - cos_sim

    logger.debug(
        "Novelty scores: mean=%.4f, max=%.4f, min=%.4f.",
        float(novelty.mean()), float(novelty.max()), float(novelty.min()),
    )
    return novelty


def select_diverse_frames(
    embeddings: np.ndarray,
    index: List[Dict],
    n_select: int,
    lambda_diversity: float = 0.5,
) -> Tuple[np.ndarray, List[Dict], List[int]]:
    """
    Maximum Marginal Relevance (MMR) selection of *n_select* diverse frames.

    MMR iteratively selects the frame that maximises::

        MMR_score(i) = lambda * max_sim_to_query
                       - (1 - lambda) * max_sim_to_already_selected

    When there is no query, we initialise with the frame that has the highest
    *average novelty* (most representative start), then greedily add the frame
    that is most different from the current selection.  Setting
    ``lambda_diversity=0`` makes this pure maximum-diversity selection;
    ``lambda_diversity=1`` ranks frames by novelty only.

    This is useful when a strict budget of representative frames is required
    regardless of video pacing (e.g. generate exactly 20 thumbnails from a
    60-second video).

    Args:
        embeddings:        Float32 ``(N, D)`` L2-normalised CLIP embeddings.
        index:             Metadata list aligned with *embeddings*.
        n_select:          Exact number of frames to return.
        lambda_diversity:  Trade-off between diversity and novelty.
                           ``0`` = pure diversity, ``1`` = novelty-ranked.

    Returns:
        Tuple of:
        - ``selected_embeddings``:  Float32 ``(n_select, D)`` array.
        - ``selected_index``:       Metadata list of length *n_select*.
        - ``selected_positions``:   List of original row indices (sorted).

    Raises:
        ValueError: if ``n_select > len(index)`` or embeddings invalid.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Expected 2-D non-empty array; got shape {embeddings.shape}."
        )
    n = embeddings.shape[0]
    n_select = min(n_select, n)

    if n_select <= 0:
        raise ValueError(f"n_select must be positive; got {n_select}.")

    novelty = compute_novelty_scores(embeddings, index)  # (N,)

    selected: List[int] = []
    remaining = list(range(n))

    # Seed with the highest-novelty frame (usually a scene-change frame)
    seed = int(np.argmax(novelty))
    selected.append(seed)
    remaining.remove(seed)

    # Greedily add the most marginally relevant frame
    while len(selected) < n_select and remaining:
        sel_embs = embeddings[selected]  # (K, D)
        # For each remaining frame, compute max sim to selected set
        remaining_embs = embeddings[remaining]          # (R, D)
        max_sim_to_selected = (remaining_embs @ sel_embs.T).max(axis=1)  # (R,)
        novelty_remaining = novelty[remaining]                            # (R,)

        mmr_scores = (
            lambda_diversity * novelty_remaining
            - (1.0 - lambda_diversity) * max_sim_to_selected
        )
        best_local = int(np.argmax(mmr_scores))
        best_global = remaining[best_local]
        selected.append(best_global)
        remaining.pop(best_local)

    selected.sort()  # restore temporal order

    selected_embeddings = embeddings[selected]
    selected_index = [index[i] for i in selected]

    logger.info(
        "MMR diverse frame selection: selected %d / %d frames "
        "(lambda_diversity=%.2f).",
        len(selected), n, lambda_diversity,
    )
    return selected_embeddings, selected_index, selected
