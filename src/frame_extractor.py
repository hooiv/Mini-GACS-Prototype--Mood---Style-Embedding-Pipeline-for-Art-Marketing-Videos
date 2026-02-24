"""
frame_extractor.py
------------------
Loads video files and extracts representative frames at a configurable
interval (e.g., one frame every N seconds).  Saves frames as JPEG images
and writes metadata to a CSV file so every downstream step can stay
reproducible and auditable.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.0,
    max_frames: Optional[int] = 50,
) -> List[dict]:
    """
    Extract frames from a video at a fixed time interval.

    Args:
        video_path:        Path to the source video file.
        output_dir:        Directory where extracted frames will be saved.
        interval_seconds:  How many seconds between extracted frames.
        max_frames:        Hard cap on the total number of frames extracted.
                           Pass None for no limit.

    Returns:
        A list of metadata dicts, one per extracted frame::

            {
                "video_id":   "<stem of video filename>",
                "timestamp":  <float seconds>,
                "frame_idx":  <int>,
                "file_path":  "<absolute path to saved JPEG>",
            }

    Raises:
        FileNotFoundError: if *video_path* does not exist.
        RuntimeError:      if the video cannot be opened by OpenCV.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("FPS reported as %s for %s; defaulting to 25.", fps, video_path)
        fps = 25.0

    video_id = Path(video_path).stem
    frame_interval = max(1, int(round(fps * interval_seconds)))

    metadata: List[dict] = []
    frame_number = 0
    saved_count = 0

    logger.info(
        "Extracting frames from '%s' (fps=%.2f, interval=%ds → every %d raw frames).",
        video_path, fps, interval_seconds, frame_interval,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            filename = f"{video_id}_frame{saved_count:04d}_t{timestamp:.2f}s.jpg"
            file_path = os.path.join(output_dir, filename)
            cv2.imwrite(file_path, frame)

            metadata.append(
                {
                    "video_id": video_id,
                    "timestamp": round(timestamp, 3),
                    "frame_idx": saved_count,
                    "file_path": os.path.abspath(file_path),
                }
            )
            saved_count += 1
            logger.debug("Saved frame %d → %s", saved_count, file_path)

            if max_frames is not None and saved_count >= max_frames:
                logger.info("Reached max_frames=%d; stopping early.", max_frames)
                break

        frame_number += 1

    cap.release()
    logger.info("Extracted %d frames from '%s'.", saved_count, video_path)
    return metadata


def save_metadata(metadata: List[dict], output_path: str, fmt: str = "csv") -> None:
    """
    Persist frame metadata to disk in CSV or JSON format.

    Args:
        metadata:     List of metadata dicts returned by :func:`extract_frames`.
        output_path:  Destination file path (extension is *not* auto-appended).
        fmt:          ``"csv"`` (default) or ``"json"``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if fmt == "json":
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Metadata written to %s (JSON, %d entries).", output_path, len(metadata))

    else:  # default: CSV
        if not metadata:
            logger.warning("Empty metadata; writing header-only CSV to %s.", output_path)
            fields = ["video_id", "timestamp", "frame_idx", "file_path"]
        else:
            fields = list(metadata[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metadata)
        logger.info("Metadata written to %s (CSV, %d entries).", output_path, len(metadata))


def load_metadata(metadata_path: str) -> List[dict]:
    """
    Load frame metadata previously saved by :func:`save_metadata`.

    Supports both CSV and JSON files (detected by extension).

    Args:
        metadata_path: Path to the metadata file.

    Returns:
        List of metadata dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file extension is not ``.csv`` or ``.json``.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    ext = Path(metadata_path).suffix.lower()
    if ext == ".json":
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    elif ext == ".csv":
        with open(metadata_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # Restore numeric types lost during CSV serialisation
        for row in rows:
            row["timestamp"] = float(row["timestamp"])
            row["frame_idx"] = int(row["frame_idx"])
        return rows
    else:
        raise ValueError(f"Unsupported metadata format: '{ext}'. Use '.csv' or '.json'.")


def process_video_directory(
    video_dir: str,
    frames_dir: str,
    metadata_dir: str,
    interval_seconds: float = 1.0,
    max_frames_per_video: int = 50,
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
) -> List[dict]:
    """
    Batch-process all videos found in *video_dir*.

    Extracts frames from each video, saves them under *frames_dir*, and
    writes per-video CSV metadata files under *metadata_dir*.

    Args:
        video_dir:              Directory containing source videos.
        frames_dir:             Root output directory for frame images.
        metadata_dir:           Directory where per-video CSV files are saved.
        interval_seconds:       Seconds between extracted frames.
        max_frames_per_video:   Upper bound on frames per video.
        extensions:             Tuple of accepted video file extensions.

    Returns:
        Combined list of all frame metadata dicts across all videos.
    """
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    video_files = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not video_files:
        logger.warning("No video files found in '%s'.", video_dir)
        return []

    all_metadata: List[dict] = []
    for vpath in video_files:
        video_id = Path(vpath).stem
        per_video_frames_dir = os.path.join(frames_dir, video_id)
        try:
            meta = extract_frames(
                vpath,
                per_video_frames_dir,
                interval_seconds=interval_seconds,
                max_frames=max_frames_per_video,
            )
            csv_path = os.path.join(metadata_dir, f"{video_id}_metadata.csv")
            save_metadata(meta, csv_path)
            all_metadata.extend(meta)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("Skipping '%s': %s", vpath, exc)

    # Also save a combined metadata file
    if all_metadata:
        combined_path = os.path.join(metadata_dir, "all_frames_metadata.csv")
        save_metadata(all_metadata, combined_path)
        logger.info("Combined metadata written to %s.", combined_path)

    return all_metadata
