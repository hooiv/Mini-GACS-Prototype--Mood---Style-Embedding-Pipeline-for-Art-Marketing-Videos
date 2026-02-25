"""
frame_extractor.py
------------------
Loads video files and extracts representative frames.

Two sampling strategies are provided:

1. **Uniform interval** (``extract_frames``) — extracts one frame every *N*
   seconds regardless of content.  Fast and simple; the right choice when
   videos are short or scene structure is unknown.

2. **Scene-adaptive** (``extract_frames_scene_adaptive``) — two-pass strategy:
   first extract a coarse set (default 2 fps) using pixel-level fingerprinting,
   detect scene transitions as large pixel-difference drops, then extract one
   representative keyframe per detected scene.  This yields a maximally
   representative set without the cluster-size bias introduced by slow pans and
   static shots.  No CLIP dependency — operates purely on grayscale pixel data.

Scene-adaptive sampling directly implements the gap noted in REPORT.md §5:
"integrating the detect_scene_transitions() output as the sampling guide would
yield more semantically representative frame sets."
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

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


def _pixel_fingerprint(frame_bgr: np.ndarray, size: int = 8) -> np.ndarray:
    """
    Compute a tiny pixel fingerprint for fast scene-change detection.

    Resizes the frame to *size*×*size* grayscale and returns it as a
    normalised float32 vector.  No CLIP or external model required.

    Args:
        frame_bgr:  BGR uint8 frame from OpenCV.
        size:       Fingerprint side length.  Default 8 → 64-D vector.

    Returns:
        Float32 array of shape ``(size*size,)``, L2-normalised.
    """
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    vec = small.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def extract_frames_scene_adaptive(
    video_path: str,
    output_dir: str,
    coarse_fps: float = 2.0,
    transition_threshold: float = 0.25,
    max_scenes: Optional[int] = 50,
    fingerprint_size: int = 8,
) -> List[dict]:
    """
    Extract one representative keyframe per detected visual scene.

    Two-pass strategy
    ~~~~~~~~~~~~~~~~~
    **Pass 1 — fast coarse scan (CPU, pixel-level):**
    Extract frames at *coarse_fps* and compute an 8×8 grayscale fingerprint
    (64-D vector) for each.  Detect scene boundaries as positions where the
    cosine distance to the previous frame's fingerprint exceeds
    *transition_threshold*.

    **Pass 2 — keyframe extraction:**
    For each detected scene, seek to its temporal midpoint in the original
    video and save one high-quality JPEG frame.

    Why this matters
    ~~~~~~~~~~~~~~~~
    Uniform sampling at 1 fps creates large redundant frame blocks wherever
    a video contains slow pans, static shots, or fades.  These blocks:

    * Inflate cluster sizes for slower-paced videos, biasing K-means toward
      video identity rather than visual style.
    * Create block-diagonal artifacts in the similarity matrix that mask
      genuine cross-video style similarity.
    * Inflate temporal coherence scores (consecutive-frame sims are high
      simply because frames are identical, not because the style is consistent).

    Scene-adaptive sampling avoids these problems by design: each scene
    contributes exactly one representative frame, regardless of its duration.

    Args:
        video_path:           Path to the source video file.
        output_dir:           Directory where extracted frames will be saved.
        coarse_fps:           Frame rate for the initial scene-detection pass.
                              Default 2.0 fps — fast yet sensitive to 0.5-second
                              scene changes.
        transition_threshold: Minimum cosine *distance* (1 − similarity) between
                              consecutive frame fingerprints to trigger a scene
                              boundary.  Default 0.25 (≈22° angular distance).
                              Increase for coarser scene detection.
        max_scenes:           Hard cap on the number of scenes (and thus frames)
                              extracted.  Default 50.
        fingerprint_size:     Side length of the pixel fingerprint.  Default 8
                              (64-D; fast; sufficient for scene detection).

    Returns:
        List of metadata dicts, one per extracted keyframe::

            {
                "video_id":        "<stem of video filename>",
                "timestamp":       <float seconds — midpoint of the scene>,
                "frame_idx":       <int — scene index 0, 1, 2, ...>,
                "file_path":       "<absolute path to saved JPEG>",
                "sampling_method": "scene_adaptive",
                "scene_id":        <int — same as frame_idx>,
                "scene_start_t":   <float seconds — scene start time>,
                "scene_end_t":     <float seconds — scene end time>,
            }

    Raises:
        FileNotFoundError: if *video_path* does not exist.
        RuntimeError:      if the video cannot be opened by OpenCV.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: coarse fingerprint scan — detect scene boundaries
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("FPS reported as %s for %s; defaulting to 25.", fps, video_path)
        fps = 25.0

    coarse_interval = max(1, int(round(fps / coarse_fps)))
    video_id = Path(video_path).stem

    fingerprints: List[np.ndarray] = []
    timestamps: List[float] = []

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % coarse_interval == 0:
            ts = frame_number / fps
            fp = _pixel_fingerprint(frame, size=fingerprint_size)
            fingerprints.append(fp)
            timestamps.append(ts)
        frame_number += 1
    cap.release()

    if not fingerprints:
        logger.warning("No frames captured from '%s'.", video_path)
        return []

    # Detect scene boundaries: large cosine-distance drops
    # scene_starts[i] = timestamp of the start of scene i
    scene_start_times: List[float] = [timestamps[0]]
    for i in range(1, len(fingerprints)):
        cos_sim = float(np.dot(fingerprints[i], fingerprints[i - 1]))
        distance = 1.0 - cos_sim  # cosine distance ∈ [0, 2] in general;
    # for L2-normalised uint8 grayscale thumbnails the practical range
    # is narrower (~[0, 1]), but we use the full theoretical bound in the
    # threshold docstring for correctness.
        if distance > transition_threshold:
            scene_start_times.append(timestamps[i])

    # Cap the number of scenes
    if max_scenes is not None and len(scene_start_times) > max_scenes:
        # Keep only the scenes with the biggest transitions (most distinct)
        scene_start_times = scene_start_times[:max_scenes]

    # Build scene intervals: (start, end) pairs
    total_duration = timestamps[-1] if timestamps else 0.0
    scene_intervals: List[Tuple[float, float]] = []
    for idx, start in enumerate(scene_start_times):
        end = (
            scene_start_times[idx + 1]
            if idx + 1 < len(scene_start_times)
            else total_duration
        )
        scene_intervals.append((start, end))

    logger.info(
        "Scene-adaptive pass: %d coarse frames → %d scenes detected "
        "(threshold=%.3f) in '%s'.",
        len(fingerprints), len(scene_intervals), transition_threshold, video_path,
    )

    # ------------------------------------------------------------------
    # Pass 2: extract one keyframe per scene (at midpoint)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    metadata: List[dict] = []

    for scene_id, (start_t, end_t) in enumerate(scene_intervals):
        mid_t = (start_t + end_t) / 2.0
        target_frame = int(round(mid_t * fps))

        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            # Fallback: seek to the scene start
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_t * fps)))
            ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame for scene %d in '%s'.", scene_id, video_path)
            continue

        filename = (
            f"{video_id}_scene{scene_id:04d}_t{mid_t:.2f}s.jpg"
        )
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, frame)

        metadata.append({
            "video_id":        video_id,
            "timestamp":       round(mid_t, 3),
            "frame_idx":       scene_id,
            "file_path":       os.path.abspath(file_path),
            "sampling_method": "scene_adaptive",
            "scene_id":        scene_id,
            "scene_start_t":   round(start_t, 3),
            "scene_end_t":     round(end_t, 3),
        })

    cap.release()
    logger.info(
        "Scene-adaptive extraction: %d keyframes saved from '%s'.",
        len(metadata), video_path,
    )
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
