"""
download_videos.py
------------------
Downloads 3 short public-domain / Creative-Commons-licensed videos that
serve as sample data for the Mini GACS pipeline.

Usage (standalone):
    python download_videos.py [--output-dir data/videos]

Dependencies: requests, tqdm (both in requirements.txt)
"""

import argparse
import hashlib
import logging
import os
from typing import List, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public-domain sample videos sourced from Wikimedia Commons.
# All files are licensed CC0 / public domain and are short (<15 s).
# ---------------------------------------------------------------------------
SAMPLE_VIDEOS: List[Tuple[str, str, str]] = [
    (
        "abstract_art",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/2/2b/"
        "Big_Buck_Bunny_excerpt_1.webm/"
        "Big_Buck_Bunny_excerpt_1.webm.360p.webm",
        "big_buck_bunny_excerpt.webm",
    ),
    (
        "marketing_clip",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/66/"
        "Sunflower_from_Pixar%27s_Toy_Story_4.webm/"
        "Sunflower_from_Pixar%27s_Toy_Story_4.webm.360p.webm",
        "sunflower_pixar.webm",
    ),
    (
        "nature_abstract",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/37/"
        "Big_Buck_Bunny_excerpt_2.webm/"
        "Big_Buck_Bunny_excerpt_2.webm.360p.webm",
        "big_buck_bunny_excerpt2.webm",
    ),
]

# Fallback mirrors (also CC0) in case primary URLs fail
FALLBACK_VIDEOS: List[Tuple[str, str, str]] = [
    (
        "abstract_art",
        "https://www.w3schools.com/html/mov_bbb.mp4",
        "abstract_art.mp4",
    ),
    (
        "marketing_clip",
        "https://www.w3schools.com/html/movie.mp4",
        "marketing_clip.mp4",
    ),
    (
        "nature_abstract",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "nature_abstract.mp4",
    ),
]


def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """
    Stream-download *url* to *dest_path*.

    Args:
        url:       Remote URL.
        dest_path: Local destination file.
        timeout:   Connection + read timeout in seconds.

    Returns:
        True on success, False on any error.
    """
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as fh, tqdm(
            desc=os.path.basename(dest_path),
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))
        logger.info("Downloaded %s → %s", url, dest_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to download %s: %s", url, exc)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_sample_videos(output_dir: str = "data/videos") -> List[str]:
    """
    Download the predefined sample videos into *output_dir*.

    Skips files that already exist (checks by filename).  Falls back to
    alternate URLs when primary URLs are unavailable.

    Args:
        output_dir: Directory where video files will be saved.

    Returns:
        List of successfully downloaded / already-present file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    downloaded: List[str] = []

    for video_id, url, filename in SAMPLE_VIDEOS:
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info("Already present: %s", dest)
            downloaded.append(dest)
            continue

        logger.info("Downloading %s from %s …", video_id, url)
        if download_file(url, dest):
            downloaded.append(dest)
        else:
            # Try fallback
            fallback = next(
                ((u, f) for vid, u, f in FALLBACK_VIDEOS if vid == video_id), None
            )
            if fallback:
                fb_url, fb_filename = fallback
                fb_dest = os.path.join(output_dir, fb_filename)
                logger.info("Trying fallback for %s: %s", video_id, fb_url)
                if download_file(fb_url, fb_dest):
                    downloaded.append(fb_dest)
                else:
                    logger.error("All downloads failed for video_id='%s'.", video_id)
            else:
                logger.error("No fallback for video_id='%s'.", video_id)

    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Download sample videos for Mini GACS.")
    parser.add_argument(
        "--output-dir", default="data/videos", help="Directory for downloaded videos."
    )
    args = parser.parse_args()

    paths = download_sample_videos(args.output_dir)
    print(f"\nDownloaded {len(paths)} video(s):")
    for p in paths:
        print(f"  {p}")
