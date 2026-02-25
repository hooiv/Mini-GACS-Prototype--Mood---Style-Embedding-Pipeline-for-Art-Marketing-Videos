"""
vector_store.py
---------------
Local cosine-similarity ANN vector store (FAISS-free).

Implements the §5 "Vector DB integration" gap (REPORT.md §18) without
requiring an external service (Qdrant, Pinecone, Weaviate) or a heavy native
library (FAISS).  Uses ``sklearn.neighbors.NearestNeighbors`` with
``algorithm='brute'`` and ``metric='cosine'``, which is:

* **Correct** — brute-force is exact (no approximation error).
* **Efficient** — at D=512 (CLIP ViT-B/32), brute-force outperforms tree
  indices (ball-tree, kd-tree) because the curse of dimensionality makes
  partitioning ineffective above ~20 dimensions.
* **Scalable enough** — up to ~50k embeddings on a CPU laptop before
  latency exceeds 100 ms per query.

For production scale (>100k embeddings, <10 ms SLA), the same
``EmbeddingVectorStore`` interface can be backed by FAISS by overriding
``_build_index()`` — no caller code changes required.

Architecture
~~~~~~~~~~~~
- Embeddings are stored as a contiguous ``(N, D)`` float32 NumPy array.
- Metadata is stored as a parallel ``List[Dict]``.
- The NearestNeighbors index is built **lazily** — only when dirty — so
  repeated ``add()`` + single ``search()`` incurs only one ``fit()`` call.
- ``add()`` uses ``np.vstack`` for correctness; O(N·D) per call.  For
  very high ingestion rates, swap to a pre-allocated buffer.
- Thread safety: ``threading.RLock`` serialises mutations; reads (search)
  acquire the same lock to prevent use of a stale index during re-build.
- Persistence: ``save(path)`` writes ``{path}.npz`` (compressed embeddings)
  and ``{path}.meta.json`` (metadata).  ``load(path)`` reconstructs the
  store; the index is rebuilt lazily on the first ``search()`` call.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Default number of nearest neighbours returned by search()
_DEFAULT_K: int = 5

# NearestNeighbors backend: 'brute' is exact and fastest for D > 20
_NN_ALGORITHM: str = "brute"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single nearest-neighbour hit from :meth:`EmbeddingVectorStore.search`.

    Attributes:
        idx:       Insertion-order index of this embedding within the store.
        metadata:  A **copy** of the associated metadata dict (safe to modify).
        score:     Cosine similarity in ``[-1, 1]``.  Higher = more similar.
    """

    idx: int
    metadata: Dict
    score: float


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class EmbeddingVectorStore:
    """
    Local cosine-similarity ANN store backed by sklearn ``NearestNeighbors``.

    Typical usage::

        store = EmbeddingVectorStore()
        store.add(embeddings, metadata_list)        # (N, D) float32
        results = store.search(query_vec, k=5)      # (D,) float32
        store.save("outputs/vector_store")
        store2 = EmbeddingVectorStore.load("outputs/vector_store")

    Thread-safe: multiple threads may call ``search()`` concurrently after the
    first search has built the index; ``add()`` serialises re-builds.

    Args:
        metric:  Distance metric for ``NearestNeighbors``.  Use ``"cosine"``
                 (default) for L2-normalised CLIP embeddings.
    """

    def __init__(self, metric: str = "cosine") -> None:
        self._metric = metric
        self._embeddings: Optional[np.ndarray] = None   # (N, D) float32
        self._metadata: List[Dict] = []
        self._nn: Optional[NearestNeighbors] = None
        self._dirty: bool = True
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Append embeddings and their metadata to the store.

        Args:
            embeddings:  Float32 array ``(M, D)`` — M new vectors.
            metadata:    List of M dicts aligned with *embeddings*.

        Raises:
            ValueError: if *embeddings* is not 2-D, or if its length does
                        not match *metadata*, or if D mismatches the store.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D; got shape {embeddings.shape}."
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) "
                "must have the same length."
            )

        with self._lock:
            embs = embeddings.astype(np.float32)
            if self._embeddings is None:
                self._embeddings = embs
            else:
                if embs.shape[1] != self._embeddings.shape[1]:
                    raise ValueError(
                        f"Dimension mismatch: store has D={self._embeddings.shape[1]}, "
                        f"new embeddings have D={embs.shape[1]}."
                    )
                self._embeddings = np.vstack([self._embeddings, embs])

            self._metadata.extend(metadata)
            self._dirty = True

            logger.debug(
                "VectorStore.add: +%d vectors; total=%d.", len(metadata), len(self)
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = _DEFAULT_K,
    ) -> List[SearchResult]:
        """
        Return the *k* most similar vectors to *query*.

        The index is rebuilt (lazily) the first time search is called after
        any ``add()`` — the cost is O(1) for brute-force since sklearn simply
        stores the reference array.

        Args:
            query:  1-D float32 array ``(D,)``.
            k:      Number of results.  Clamped to ``len(self)`` automatically.

        Returns:
            List of :class:`SearchResult` sorted by **descending** similarity.

        Raises:
            RuntimeError: if the store is empty.
            ValueError:   if *query* dimensionality mismatches the store, or
                          if *query* is not 1-D.
        """
        with self._lock:
            if self._embeddings is None or len(self) == 0:
                raise RuntimeError("Vector store is empty.")

            if query.ndim != 1:
                raise ValueError(
                    f"query must be 1-D; got shape {query.shape}."
                )
            if query.shape[0] != self._embeddings.shape[1]:
                raise ValueError(
                    f"query dim={query.shape[0]} mismatches store dim="
                    f"{self._embeddings.shape[1]}."
                )

            if self._dirty:
                self._build_index()

            k_actual = min(k, len(self))
            q = query.astype(np.float32).reshape(1, -1)

            # NearestNeighbors returns cosine *distance* ∈ [0, 2].
            # Cosine similarity = 1 − distance.
            distances, indices = self._nn.kneighbors(q, n_neighbors=k_actual)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                results.append(
                    SearchResult(
                        idx=int(idx),
                        metadata=dict(self._metadata[idx]),
                        score=float(np.clip(1.0 - dist, -1.0, 1.0)),
                    )
                )

            logger.debug(
                "VectorStore.search: k=%d, top_score=%.4f.",
                k_actual,
                results[0].score if results else float("nan"),
            )
            return results

    def _build_index(self) -> None:
        """Build or rebuild the NearestNeighbors index (call inside lock)."""
        n = len(self)
        self._nn = NearestNeighbors(
            n_neighbors=min(n, _DEFAULT_K),
            algorithm=_NN_ALGORITHM,
            metric=self._metric,
            n_jobs=1,
        )
        self._nn.fit(self._embeddings)
        self._dirty = False
        logger.debug(
            "VectorStore: index built for %d vectors (D=%d).",
            n,
            self._embeddings.shape[1],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the store to ``{path}.npz`` (embeddings, compressed) and
        ``{path}.meta.json`` (metadata).

        Args:
            path:  Base path **without** extension.

        Raises:
            RuntimeError: if the store is empty.
        """
        with self._lock:
            if self._embeddings is None:
                raise RuntimeError("Cannot save an empty vector store.")

            emb_path = f"{path}.npz"
            meta_path = f"{path}.meta.json"

            np.savez_compressed(emb_path, embeddings=self._embeddings)
            with open(meta_path, "w", encoding="utf-8") as fp:
                json.dump(self._metadata, fp)

            logger.info(
                "VectorStore saved: %d vectors to %s.", len(self), emb_path
            )

    @classmethod
    def load(cls, path: str) -> "EmbeddingVectorStore":
        """
        Load a previously saved store.

        Args:
            path:  Base path used in :meth:`save` (without extension).

        Returns:
            Populated :class:`EmbeddingVectorStore`.  The index is rebuilt
            lazily on the first :meth:`search` call.

        Raises:
            FileNotFoundError: if either expected file is missing.
        """
        emb_path = f"{path}.npz"
        meta_path = f"{path}.meta.json"

        for p in (emb_path, meta_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Vector store file not found: {p!r}"
                )

        data = np.load(emb_path)
        embeddings = data["embeddings"].astype(np.float32)
        with open(meta_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        store = cls()
        store._embeddings = embeddings
        store._metadata = metadata
        store._dirty = True  # rebuild index on first search

        logger.info(
            "VectorStore loaded: %d vectors from %s.", len(store), emb_path
        )
        return store

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[0]

    def __repr__(self) -> str:
        dim = "?" if self._embeddings is None else self._embeddings.shape[1]
        return (
            f"EmbeddingVectorStore("
            f"n={len(self)}, D={dim}, metric={self._metric!r})"
        )
