"""
similarity.py
-------------
Computes pairwise cosine similarity between frame embeddings and provides
top-k nearest-neighbour retrieval.  All embeddings are assumed to be
L2-normalised (as produced by ``embeddings.py``), so cosine similarity
equals the dot product.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine-similarity matrix.

    Because embeddings are L2-normalised the similarity is just ``E @ E.T``.

    Args:
        embeddings: Float32 array ``(N, D)`` of L2-normalised vectors.

    Returns:
        Float32 symmetric matrix ``(N, N)`` with values in ``[-1, 1]``.

    Raises:
        ValueError: if *embeddings* is empty or not 2-D.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Expected a 2-D non-empty array; got shape {embeddings.shape}."
        )

    sim = (embeddings @ embeddings.T).astype(np.float32)

    # Clip to valid range to guard against tiny floating-point overflows
    sim = np.clip(sim, -1.0, 1.0)

    # Sanity checks
    if np.isnan(sim).any():
        raise ValueError("Similarity matrix contains NaN values.")
    if np.isinf(sim).any():
        raise ValueError("Similarity matrix contains Inf values.")

    logger.debug(
        "Similarity matrix computed: shape=%s, range=[%.4f, %.4f].",
        sim.shape, float(sim.min()), float(sim.max()),
    )
    return sim


def get_top_k_similar(
    query_idx: int,
    similarity_matrix: np.ndarray,
    index: List[Dict],
    top_k: int = 5,
    exclude_self: bool = True,
) -> List[Dict]:
    """
    Retrieve the *top_k* most similar frames for a given query frame.

    Args:
        query_idx:          Row index in *similarity_matrix* for the query.
        similarity_matrix:  Precomputed ``(N, N)`` similarity matrix.
        index:              List of metadata dicts aligned with the rows of
                            *similarity_matrix*.
        top_k:              Number of results to return.
        exclude_self:       If True, the query frame itself is excluded from
                            results even if it is the top hit.

    Returns:
        List of at most *top_k* dicts, each being the original metadata entry
        augmented with a ``"similarity"`` key (float).

    Raises:
        IndexError: if *query_idx* is out of bounds.
    """
    n = similarity_matrix.shape[0]
    if not (0 <= query_idx < n):
        raise IndexError(
            f"query_idx {query_idx} out of range for similarity matrix of size {n}."
        )

    scores = similarity_matrix[query_idx].copy()

    if exclude_self:
        scores[query_idx] = -np.inf  # push self to the back

    # Descending sort
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:top_k]

    results = []
    for idx in top_indices:
        if idx >= len(index):
            continue
        entry = dict(index[idx])
        entry["similarity"] = float(scores[idx])
        results.append(entry)

    logger.debug(
        "Top-%d results for query_idx=%d: similarities=%s.",
        top_k, query_idx, [round(r["similarity"], 4) for r in results],
    )
    return results


def batch_top_k_queries(
    query_indices: List[int],
    similarity_matrix: np.ndarray,
    index: List[Dict],
    top_k: int = 5,
) -> Dict[int, List[Dict]]:
    """
    Run :func:`get_top_k_similar` for multiple query indices.

    Args:
        query_indices:      List of row indices to use as queries.
        similarity_matrix:  Precomputed ``(N, N)`` similarity matrix.
        index:              Metadata list aligned with *similarity_matrix*.
        top_k:              Number of results per query.

    Returns:
        Dict mapping each query index to its list of top-k result dicts.
    """
    results: Dict[int, List[Dict]] = {}
    for qidx in query_indices:
        try:
            results[qidx] = get_top_k_similar(qidx, similarity_matrix, index, top_k)
        except (IndexError, ValueError) as exc:
            logger.error("Skipping query_idx=%d: %s", qidx, exc)
    return results


def compute_inter_video_stats(
    similarity_matrix: np.ndarray,
    index: List[Dict],
) -> Dict[str, float]:
    """
    Summarise mean and median pairwise similarities, overall and split by
    *within-video* vs *cross-video* frame pairs.

    Uses vectorised NumPy boolean indexing over the upper triangle of the
    similarity matrix — no Python loop over frame pairs.

    Args:
        similarity_matrix: ``(N, N)`` cosine-similarity matrix.
        index:             Metadata list aligned with rows.

    Returns:
        Dict with keys:
        - ``"overall_mean"``
        - ``"overall_median"``
        - ``"within_video_mean"``
        - ``"cross_video_mean"``
    """
    n = similarity_matrix.shape[0]
    video_ids = np.array([entry.get("video_id", "") for entry in index])

    # Upper-triangle mask (excludes diagonal and lower triangle)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Within-video pairs: same video_id in the upper triangle
    same_video = video_ids[:, None] == video_ids[None, :]  # (N, N) bool
    within_mask = same_video & upper
    cross_mask = (~same_video) & upper

    within_vals = similarity_matrix[within_mask]
    cross_vals = similarity_matrix[cross_mask]
    all_vals = similarity_matrix[upper]

    stats = {
        "overall_mean": float(np.mean(all_vals)) if all_vals.size else float("nan"),
        "overall_median": float(np.median(all_vals)) if all_vals.size else float("nan"),
        "within_video_mean": float(np.mean(within_vals)) if within_vals.size else float("nan"),
        "cross_video_mean": float(np.mean(cross_vals)) if cross_vals.size else float("nan"),
    }
    logger.info("Inter-video similarity stats: %s", stats)
    return stats
