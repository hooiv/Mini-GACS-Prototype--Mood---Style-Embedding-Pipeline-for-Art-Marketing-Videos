"""
experiment_manifest.py
----------------------
Structured run manifest for the Mini GACS pipeline.

Every pipeline run produces a single ``run_manifest.json`` that records:
- A unique run ID (ISO-8601 UTC timestamp)
- Per-module metrics (embedding shape, cluster quality, Spearman ρ, …)
- All output artifact paths + file sizes
- The key hyperparameters used in this run

This enables reproducibility auditing: two engineers can compare
``run_manifest.json`` files from different runs to understand what changed
and why results differ.  It is also the foundation for an automated
experiment-tracking dashboard (MLflow / W&B integration is a one-line
adapter from this format).

Usage
-----
    from src.experiment_manifest import PipelineManifest

    manifest = PipelineManifest()
    manifest.record("frame_extraction", n_videos=3, n_frames=72)
    manifest.record("embeddings", shape=[72, 512], model="clip-vit-base-patch32")
    manifest.record("deduplication", n_before=72, n_after=45, threshold=0.97)
    manifest.record("clustering", n_clusters=5, silhouette=0.42)
    manifest.add_artifact("outputs/similarity_heatmap.png", "Cosine sim heatmap")
    manifest.save("outputs/run_manifest.json")
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineManifest:
    """
    Accumulates metrics and artifact paths across a pipeline run and
    serialises them to a single JSON file at the end.

    Args:
        run_id:  Optional explicit run identifier.  Defaults to an ISO-8601
                 UTC timestamp (e.g. ``"20260224T153045"``) which is unique
                 to the second and human-readable.
        config:  Optional dict of top-level hyperparameters (e.g. model
                 name, frame interval, dedup threshold) to embed in the
                 manifest for full reproducibility.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id: str = run_id or datetime.now(tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )
        self._data: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "config": config or {},
            "modules": {},
            "artifacts": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, module_name: str, **metrics: Any) -> None:
        """
        Record metrics emitted by one pipeline module.

        All keyword arguments are stored as-is.  NumPy scalar types are
        automatically cast to Python native types for JSON serialisability.

        Args:
            module_name:  Short name for the pipeline step (e.g.
                          ``"frame_extraction"``, ``"clustering"``).
            **metrics:    Arbitrary key-value pairs.  Values must be
                          JSON-serialisable (str, int, float, list, dict).

        Example::

            manifest.record(
                "clustering",
                n_clusters=5,
                silhouette=0.42,
                davies_bouldin=1.1,
                inertia=34.7,
            )
        """
        sanitised = {k: _to_json_serialisable(v) for k, v in metrics.items()}
        sanitised["_recorded_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._data["modules"][module_name] = sanitised
        logger.debug("Manifest: recorded module '%s' with %d metrics.", module_name, len(metrics))

    def add_artifact(self, path: str, description: str) -> None:
        """
        Register an output artifact (file path) with a human-readable description.

        The file size is recorded if the file already exists; otherwise
        ``null`` is stored (useful for files written after this call).

        Args:
            path:         Absolute or relative path to the artifact file.
            description:  One-sentence description of what the file contains.
        """
        abs_path = os.path.abspath(path)
        size = os.path.getsize(abs_path) if os.path.exists(abs_path) else None
        self._data["artifacts"].append(
            {
                "path": abs_path,
                "description": description,
                "size_bytes": size,
            }
        )

    def get(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Return the recorded metrics for *module_name*, or None."""
        return self._data["modules"].get(module_name)

    def save(self, output_path: str) -> str:
        """
        Write the manifest to a JSON file.

        Args:
            output_path:  Destination file path.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        logger.info("Run manifest saved to %s (run_id=%s).", output_path, self.run_id)
        return os.path.abspath(output_path)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of recorded metrics."""
        lines = [f"Run ID: {self.run_id}"]
        for module, metrics in self._data["modules"].items():
            lines.append(f"  [{module}]")
            for k, v in metrics.items():
                if k.startswith("_"):
                    continue
                lines.append(f"    {k}: {v}")
        n_artifacts = len(self._data["artifacts"])
        lines.append(f"  Artifacts: {n_artifacts} file(s) registered")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_serialisable(value: Any) -> Any:
    """
    Recursively convert NumPy scalars / arrays and other non-JSON types
    to Python native types.
    """
    try:
        import numpy as np  # optional for the module
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if isinstance(value, dict):
        return {k: _to_json_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serialisable(v) for v in value]
    return value
