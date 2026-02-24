"""
clustering.py
-------------
Clusters frame embeddings by visual vibe using K-means and visualises the
result with a 2-D PCA or t-SNE projection.

This module realises the "cluster a large creative library by mood" concept
described in REPORT.md §2b.  In practice, creative teams use such a map to:

* Spot visual style groups across a whole campaign library at a glance.
* Quickly find frames that visually diverge from the rest of a campaign.
* Seed a vector-DB with canonical per-cluster representatives.

Usage
-----
    from src.clustering import VibeClusterer

    clust = VibeClusterer(n_clusters=5)
    labels = clust.fit(embeddings)
    clust.plot_scatter(
        embeddings, labels, index,
        output_path="outputs/vibe_cluster_scatter.png",
    )
    clust.save_cluster_assignments(labels, index, "outputs/cluster_assignments.json")
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class VibeClusterer:
    """
    Groups frame embeddings into *n_clusters* visual-vibe clusters using
    K-means and provides 2-D projection helpers for visualisation.

    Args:
        n_clusters:   Number of clusters (default 5).
        random_state: Random seed for reproducibility.
        n_init:       Number of K-means initialisations (default 10).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be ≥ 2; got {n_clusters}.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
        self._pca: Optional[PCA] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit K-means on *embeddings* and return per-frame cluster labels.

        Args:
            embeddings:  Float32 ``(N, D)`` array of L2-normalised frame
                         embeddings.  *N* must be ≥ *n_clusters*.

        Returns:
            Integer array of shape ``(N,)`` with values in
            ``[0, n_clusters - 1]``.

        Raises:
            ValueError: if fewer frames than clusters are provided.
        """
        self._validate_embeddings(embeddings)
        n = embeddings.shape[0]
        if n < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} frames to form {self.n_clusters} "
                f"clusters; got {n}."
            )

        labels = self._kmeans.fit_predict(embeddings).astype(np.int32)
        self._is_fitted = True
        logger.info(
            "K-means fitted: %d frames → %d clusters. "
            "Inertia=%.4f.",
            n, self.n_clusters, float(self._kmeans.inertia_),
        )
        return labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign new embeddings to the nearest cluster centre.

        Args:
            embeddings:  ``(M, D)`` float32 array.

        Returns:
            Integer label array of shape ``(M,)``.

        Raises:
            RuntimeError: if :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        self._validate_embeddings(embeddings)
        return self._kmeans.predict(embeddings).astype(np.int32)

    def cluster_summary(
        self,
        labels: np.ndarray,
        index: List[Dict],
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict]:
        """
        Summarise each cluster: size, video-ID distribution, and which
        frame index is closest to the cluster centroid.

        Args:
            labels:      Per-frame cluster labels from :meth:`fit`.
            index:       Metadata list aligned with *labels*.
            embeddings:  Optional ``(N, D)`` frame embeddings used to find
                         the frame nearest to each centroid.  When *None*,
                         the first frame in each cluster is used instead.

        Returns:
            Dict ``{cluster_id: {size, video_distribution, representative_frame_idx}}``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before cluster_summary().")

        summary: Dict[int, Dict] = {}

        for k in range(self.n_clusters):
            mask = labels == k
            cluster_indices = np.where(mask)[0]
            cluster_meta = [index[i] for i in cluster_indices if i < len(index)]

            # Video distribution within the cluster
            vid_counts: Dict[str, int] = {}
            for entry in cluster_meta:
                vid = entry.get("video_id", "unknown")
                vid_counts[vid] = vid_counts.get(vid, 0) + 1

            # Frame closest to the cluster centroid (representative frame)
            representative_frame_idx: Optional[int] = None
            if len(cluster_indices) > 0:
                if embeddings is not None:
                    centroid = self._kmeans.cluster_centers_[k]  # (D,)
                    cluster_embs = embeddings[cluster_indices]    # (M, D)
                    distances = np.linalg.norm(cluster_embs - centroid, axis=1)
                    representative_frame_idx = int(cluster_indices[int(np.argmin(distances))])
                else:
                    representative_frame_idx = int(cluster_indices[0])

            summary[k] = {
                "size": int(mask.sum()),
                "video_distribution": vid_counts,
                "representative_frame_idx": representative_frame_idx,
            }

        return summary

    # ------------------------------------------------------------------
    # Dimensionality reduction helpers
    # ------------------------------------------------------------------

    def project_2d(
        self,
        embeddings: np.ndarray,
        method: str = "pca",
    ) -> np.ndarray:
        """
        Project high-dimensional embeddings to 2-D for visualisation.

        Args:
            embeddings:  ``(N, D)`` float32 array.
            method:      ``"pca"`` (default) or ``"tsne"``.

        Returns:
            Float32 array ``(N, 2)``.

        Raises:
            ValueError: if *method* is unknown or *embeddings* is 1-D.
        """
        self._validate_embeddings(embeddings)
        n, d = embeddings.shape

        if d <= 2:
            # Already low-dimensional; pad or return as-is
            if d == 1:
                return np.hstack([embeddings, np.zeros((n, 1), dtype=np.float32)])
            return embeddings.astype(np.float32)

        if method == "pca":
            n_components = min(2, n, d)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            coords = pca.fit_transform(embeddings).astype(np.float32)
            if coords.shape[1] < 2:
                # Pad with zeros if only 1 PC was possible
                coords = np.hstack(
                    [coords, np.zeros((n, 2 - coords.shape[1]), dtype=np.float32)]
                )
            self._pca = pca
            logger.debug(
                "PCA projection: explained variance ratio = %s.",
                pca.explained_variance_ratio_,
            )
            return coords

        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
            except ImportError as exc:
                raise ImportError("scikit-learn is required for t-SNE.") from exc

            perplexity = min(30, max(5, n // 3))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=self.random_state,
                n_iter=1000,
            )
            return tsne.fit_transform(embeddings).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown projection method '{method}'. Use 'pca' or 'tsne'."
            )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_scatter(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        index: List[Dict],
        output_path: str,
        projection: str = "pca",
        title: str = "Vibe Cluster Map (PCA projection)",
        figsize: Tuple[int, int] = (10, 8),
        annotate_videos: bool = True,
    ) -> str:
        """
        Plot a 2-D scatter of frame embeddings coloured by cluster.

        Each point represents one frame.  Points are colour-coded by cluster
        label; distinct video sources are shown with different marker shapes.

        Args:
            embeddings:      ``(N, D)`` L2-normalised frame embeddings.
            labels:          Cluster labels from :meth:`fit`, shape ``(N,)``.
            index:           Frame metadata list aligned with rows.
            output_path:     Destination PNG file.
            projection:      ``"pca"`` (default) or ``"tsne"``.
            title:           Figure title.
            figsize:         ``(width, height)`` in inches.
            annotate_videos: If True, add video-ID annotations to cluster
                             centroids in the 2-D space.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        coords = self.project_2d(embeddings, method=projection)

        # Marker shapes cycle over unique video IDs
        video_ids = [entry.get("video_id", "unknown") for entry in index]
        unique_vids = sorted(set(video_ids))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        vid_to_marker = {v: markers[i % len(markers)] for i, v in enumerate(unique_vids)}

        # Colour palette for clusters
        cmap = plt.cm.tab10
        colours = [cmap(k / max(self.n_clusters - 1, 1)) for k in range(self.n_clusters)]

        fig, ax = plt.subplots(figsize=figsize)

        for vid in unique_vids:
            vid_mask = np.array([v == vid for v in video_ids])
            for k in range(self.n_clusters):
                mask = vid_mask & (labels == k)
                if not mask.any():
                    continue
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[colours[k]],
                    marker=vid_to_marker[vid],
                    s=60,
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.4,
                    label=f"cluster {k} / {vid}" if vid == unique_vids[0] else None,
                )

        # Annotate cluster centroids in 2-D space
        if annotate_videos and self._is_fitted:
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    cx = float(coords[mask, 0].mean())
                    cy = float(coords[mask, 1].mean())
                    ax.annotate(
                        f" C{k}",
                        (cx, cy),
                        fontsize=11,
                        fontweight="bold",
                        color=colours[k],
                        ha="center",
                    )

        # Build a tidy legend (one entry per cluster only)
        cluster_handles = [
            plt.scatter([], [], c=[colours[k]], marker="o", s=60,
                        label=f"Cluster {k}")
            for k in range(self.n_clusters)
        ]
        vid_handles = [
            plt.scatter([], [], c=["grey"], marker=vid_to_marker[v], s=60,
                        label=v)
            for v in unique_vids
        ]
        ax.legend(
            handles=cluster_handles + vid_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.7,
        )

        ax.set_xlabel(f"{projection.upper()} dim-1", fontsize=10)
        ax.set_ylabel(f"{projection.upper()} dim-2", fontsize=10)
        ax.set_title(title, fontsize=13, pad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Cluster scatter saved to %s.", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_cluster_assignments(
        self,
        labels: np.ndarray,
        index: List[Dict],
        output_path: str,
    ) -> str:
        """
        Save per-frame cluster assignments to a JSON file.

        Format::

            [{"video_id": ..., "frame_idx": ..., "cluster": 2, ...}, ...]

        Args:
            labels:       Cluster label array ``(N,)``.
            index:        Frame metadata list aligned with *labels*.
            output_path:  Destination JSON file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rows = []
        for i, entry in enumerate(index):
            row = dict(entry)
            row["cluster"] = int(labels[i]) if i < len(labels) else -1
            rows.append(row)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

        logger.info(
            "Cluster assignments saved to %s (%d rows).", output_path, len(rows)
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Expected 2-D non-empty array; got shape {embeddings.shape}."
            )
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values.")


def auto_n_clusters(embeddings: np.ndarray, max_k: int = 10) -> int:
    """
    Suggest a number of clusters using the elbow heuristic (largest drop in
    inertia per added cluster).

    Args:
        embeddings:  ``(N, D)`` float32 L2-normalised embeddings.
        max_k:       Maximum *k* to consider.

    Returns:
        Suggested integer *k* in ``[2, max_k]``.
    """
    n = embeddings.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2

    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(embeddings)
        inertias.append(km.inertia_)

    # Largest relative drop
    drops = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
    if not drops:
        # Only one k was tested – return it directly
        return ks[0]
    best_idx = int(np.argmax(drops))
    suggested_k = ks[best_idx + 1]  # k that caused the largest drop

    logger.info(
        "auto_n_clusters: tested k=%s, inertias=%s → suggested k=%d.",
        ks, [round(x, 1) for x in inertias], suggested_k,
    )
    return suggested_k
