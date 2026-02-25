"""
pipeline_config.py
------------------
Typed, validated, serialisable configuration for the Mini-GACS pipeline.

Replaces the raw ``argparse.Namespace`` used throughout ``main.py`` with a
:class:`PipelineConfig` dataclass that provides:

* **Type safety** — every field has an explicit Python type annotation.
* **Validation** — ``__post_init__`` raises ``ValueError`` for nonsensical
  parameter combinations (e.g. ``fps <= 0``, ``n_clusters < 0``).
* **Persistence** — :meth:`to_dict` / :meth:`from_dict` / :meth:`save` /
  :meth:`load` for reproducible experiment tracking; the config is recorded
  in the run manifest so any run can be exactly reproduced.
* **Backward compat** — :meth:`from_args` converts an ``argparse.Namespace``
  to a :class:`PipelineConfig` so existing call-sites work unchanged.

Why this matters
----------------
Without a typed configuration object:

1. Parameters like ``args.interval`` and ``args.model`` are bare strings/
   floats that can silently be passed to the wrong function.
2. There is no single source of truth for defaults — ``parse_args()`` in
   ``main.py`` and every module that has its own default value can diverge.
3. Saving the exact parameters used for a run requires manually constructing
   a dict from the ``Namespace`` at multiple call-sites, creating
   boilerplate and drift risk.

Usage
-----
    from src.pipeline_config import PipelineConfig

    # Construct with explicit values (other fields take defaults)
    cfg = PipelineConfig(fps=2.0, n_clusters=5)

    # Validate-then-save for the manifest
    cfg.save("outputs/run_config.json")

    # Reload in a subsequent analysis session
    cfg2 = PipelineConfig.load("outputs/run_config.json")
    assert cfg2.fps == cfg.fps

    # Backward-compat: convert from argparse
    import argparse
    ns = argparse.Namespace(interval=1.0, model="openai/clip-vit-base-patch32")
    cfg3 = PipelineConfig.from_args(ns)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values — single source of truth shared across all modules
# ---------------------------------------------------------------------------
_DEFAULT_VIDEO_DIR      = "data/videos"
_DEFAULT_FRAME_DIR      = "data/frames"
_DEFAULT_EMBED_DIR      = "data/embeddings"
_DEFAULT_META_DIR       = "data/metadata"
_DEFAULT_OUTPUT_DIR     = "outputs"
_DEFAULT_MODEL_NAME     = "openai/clip-vit-base-patch32"
_DEFAULT_FPS            = 1.0
_DEFAULT_MAX_FRAMES     = 50
_DEFAULT_TOP_K          = 5
_DEFAULT_N_QUERIES      = 3
_DEFAULT_N_CLUSTERS     = 0       # 0 = auto via silhouette
_DEFAULT_DEDUP_THRESH   = 0.97
_DEFAULT_MIN_QUALITY    = 0.25
_DEFAULT_BOOTSTRAP_N    = 1000
_DEFAULT_SCENE_ADAPTIVE = False


@dataclass
class PipelineConfig:
    """
    Full configuration for the 16-step Mini-GACS pipeline.

    All fields have sensible defaults; only override what you need to change.

    Attributes
    ----------
    video_dir:
        Directory containing input video files (.mp4/.avi/.mov/.mkv/.webm).
    frame_dir:
        Directory where extracted frames (JPEG/PNG) are saved.
    embed_dir:
        Directory where ``.npy`` + ``.json`` embedding files are written.
    meta_dir:
        Directory for CSV/JSON frame-level metadata.
    output_dir:
        Root directory for visualisations, reports, and run artifacts.
    model_name:
        HuggingFace CLIP checkpoint identifier (must match between extraction
        and retrieval sessions to ensure embedding compatibility).
    fps:
        Frames per second for uniform-interval extraction.  Ignored when
        ``scene_adaptive=True``.
    max_frames:
        Maximum frames to retain per video after extraction (further capped
        by deduplication and quality filtering).
    top_k:
        Number of nearest neighbours to retrieve per query frame.
    n_queries:
        Number of query frames selected for the top-k retrieval step.
    n_clusters:
        Number of K-means vibe clusters.  ``0`` enables automatic selection
        via silhouette score.
    dedup_threshold:
        Cosine-similarity threshold for near-duplicate frame removal
        (higher = stricter; set to ``1.0`` to disable deduplication).
    min_quality_score:
        Minimum composite technical-quality score ``[0, 1]`` for a frame to
        pass the quality filter.  Set to ``0.0`` to disable filtering.
    bootstrap_n:
        Bootstrap resamples used to compute 95% CIs in creative ranking.
        Must be ≥ 100 for reliable intervals.
    scene_adaptive:
        Use pixel-fingerprint scene-adaptive sampling instead of uniform
        interval sampling.
    skip_download:
        Skip the video download step (re-use files already in ``video_dir``).
    skip_extraction:
        Skip frame extraction (re-use frames and metadata already on disk).
    skip_embedding:
        Skip embedding computation (re-use saved ``.npy``/``.json`` files).
    skip_similarity:
        Skip the similarity matrix computation step.
    """

    # I/O paths
    video_dir:   str = _DEFAULT_VIDEO_DIR
    frame_dir:   str = _DEFAULT_FRAME_DIR
    embed_dir:   str = _DEFAULT_EMBED_DIR
    meta_dir:    str = _DEFAULT_META_DIR
    output_dir:  str = _DEFAULT_OUTPUT_DIR

    # Model
    model_name: str = _DEFAULT_MODEL_NAME

    # Extraction parameters
    fps:            float = _DEFAULT_FPS
    max_frames:     int   = _DEFAULT_MAX_FRAMES
    scene_adaptive: bool  = _DEFAULT_SCENE_ADAPTIVE

    # Retrieval / clustering
    top_k:      int = _DEFAULT_TOP_K
    n_queries:  int = _DEFAULT_N_QUERIES
    n_clusters: int = _DEFAULT_N_CLUSTERS

    # Quality-control thresholds
    dedup_threshold:   float = _DEFAULT_DEDUP_THRESH
    min_quality_score: float = _DEFAULT_MIN_QUALITY

    # Statistical parameters
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N

    # Skip-step flags
    skip_download:   bool = False
    skip_extraction: bool = False
    skip_embedding:  bool = False
    skip_similarity: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all field values on construction."""
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0; got {self.fps}.")
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1; got {self.max_frames}.")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}.")
        if self.n_queries < 1:
            raise ValueError(f"n_queries must be >= 1; got {self.n_queries}.")
        if self.n_clusters < 0:
            raise ValueError(
                f"n_clusters must be >= 0 (0 = auto); got {self.n_clusters}."
            )
        if not 0.0 <= self.dedup_threshold <= 1.0:
            raise ValueError(
                f"dedup_threshold must be in [0, 1]; got {self.dedup_threshold}."
            )
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                f"min_quality_score must be in [0, 1]; got {self.min_quality_score}."
            )
        if self.bootstrap_n < 100:
            raise ValueError(
                f"bootstrap_n must be >= 100 for reliable CIs; "
                f"got {self.bootstrap_n}."
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation (JSON-serialisable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """
        Construct from a plain dict (e.g., parsed from JSON).

        Unknown keys are silently ignored for forward-compatibility.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """
        Serialise to a JSON file.

        Args:
            path:  Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("PipelineConfig saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """
        Load from a JSON file written by :meth:`save`.

        Args:
            path:  Source file path.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: '{path}'.")
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cfg = cls.from_dict(d)
        logger.info("PipelineConfig loaded from '%s'.", path)
        return cfg

    # ------------------------------------------------------------------
    # Backward-compat adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args: Any) -> "PipelineConfig":
        """
        Build a :class:`PipelineConfig` from an ``argparse.Namespace``.

        Provides backward compatibility with the existing ``parse_args()``
        in ``main.py``.  Only recognised attribute names are transferred;
        unknown argparse attributes are silently ignored.

        The following argparse name mappings are applied:

        * ``args.interval`` → ``fps``  (argparse uses ``--interval`` for the
          frame-extraction interval in seconds)
        * ``args.model``    → ``model_name``
        * Everything else uses the same name in both systems.
        """
        # argparse attribute name → PipelineConfig field name
        _MAPPING = {
            "interval":          "fps",
            "model":             "model_name",
            "max_frames":        "max_frames",
            "top_k":             "top_k",
            "n_queries":         "n_queries",
            "n_clusters":        "n_clusters",
            "dedup_threshold":   "dedup_threshold",
            "min_quality_score": "min_quality_score",
            "bootstrap_n":       "bootstrap_n",
            "scene_adaptive":    "scene_adaptive",
            "skip_download":     "skip_download",
            "skip_extraction":   "skip_extraction",
            "skip_embedding":    "skip_embedding",
            "skip_similarity":   "skip_similarity",
        }
        kwargs: Dict[str, Any] = {}
        for arg_attr, cfg_field in _MAPPING.items():
            val = getattr(args, arg_attr, None)
            if val is not None:
                kwargs[cfg_field] = val
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["PipelineConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
