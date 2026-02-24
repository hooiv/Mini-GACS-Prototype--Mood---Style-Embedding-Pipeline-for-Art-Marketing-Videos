"""
main.py
-------
End-to-end pipeline orchestrator for the Mini GACS
Mood & Style Embedding Pipeline.

Steps
-----
1. (Optional) Download sample videos.
2. Extract frames from all videos found in ``data/videos/``.
3. Compute CLIP embeddings for all frames.
4. Calculate pairwise cosine-similarity matrix.
5. Run top-5 retrieval for at least 3 query frames.
6. Generate visualisations (heatmap, bar chart, retrieval grids).
7. Write a Markdown similarity report.

Usage
-----
    # Full pipeline (downloads videos if directory is empty):
    python main.py

    # Skip download step if you already have videos:
    python main.py --skip-download

    # Override interval / model:
    python main.py --interval 2 --model openai/clip-vit-base-patch32
"""

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Configure logging before any local imports
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.frame_extractor import process_video_directory, load_metadata
from src.embeddings import compute_and_save_embeddings, load_embeddings
from src.similarity import (
    cosine_similarity_matrix,
    batch_top_k_queries,
    compute_inter_video_stats,
)
from src.visualization import (
    plot_similarity_heatmap,
    plot_cross_video_similarity_bar,
    plot_top_k_grid,
    generate_similarity_report,
)
from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
OUTPUTS_DIR = "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mini GACS – Mood & Style Embedding Pipeline"
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not attempt to download sample videos.",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Re-use previously extracted frames (metadata must exist).",
    )
    p.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Re-use previously computed embeddings.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds between extracted frames (default: 1.0).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=50,
        metavar="N",
        help="Maximum frames per video (default: 50).",
    )
    p.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model identifier.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar frames to retrieve per query (default: 5).",
    )
    p.add_argument(
        "--n-queries",
        type=int,
        default=3,
        help="Number of query frames selected for top-k retrieval (default: 3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Step 1 – Download sample videos
    # -----------------------------------------------------------------------
    if not args.skip_download:
        logger.info("=== Step 1: Download sample videos ===")
        try:
            from download_videos import download_sample_videos
            downloaded = download_sample_videos(VIDEOS_DIR)
            logger.info("Videos available: %d", len(downloaded))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Video download failed (%s); continuing anyway.", exc)

    # -----------------------------------------------------------------------
    # Step 2 – Extract frames
    # -----------------------------------------------------------------------
    combined_csv = os.path.join(METADATA_DIR, "all_frames_metadata.csv")

    if args.skip_extraction and os.path.exists(combined_csv):
        logger.info("=== Step 2: Loading existing frame metadata ===")
        metadata = load_metadata(combined_csv)
    else:
        logger.info("=== Step 2: Extract frames from videos ===")
        metadata = process_video_directory(
            video_dir=VIDEOS_DIR,
            frames_dir=FRAMES_DIR,
            metadata_dir=METADATA_DIR,
            interval_seconds=args.interval,
            max_frames_per_video=args.max_frames,
        )

    if not metadata:
        logger.error(
            "No frames available.  Place video files in '%s' and re-run.", VIDEOS_DIR
        )
        sys.exit(1)

    logger.info("Total frames: %d", len(metadata))

    # Print a quick preview
    print("\n── Frame metadata preview (first 3 rows) ──")
    for row in metadata[:3]:
        print(f"  {row}")
    print()

    # -----------------------------------------------------------------------
    # Step 3 – Compute embeddings
    # -----------------------------------------------------------------------
    npy_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")

    if args.skip_embedding and os.path.exists(npy_path):
        logger.info("=== Step 3: Loading existing embeddings ===")
        embeddings, index = load_embeddings(EMBEDDINGS_DIR)
    else:
        logger.info("=== Step 3: Compute CLIP embeddings ===")
        embeddings, index = compute_and_save_embeddings(
            metadata=metadata,
            output_dir=EMBEDDINGS_DIR,
            model_name=args.model,
        )

    print(f"\n── Embeddings shape: {embeddings.shape}  (dtype={embeddings.dtype}) ──\n")

    # Sanity assertions
    assert embeddings.ndim == 2, "Embeddings must be 2-D."
    assert not (embeddings != embeddings).any(), "Embeddings must not contain NaN."
    assert len(index) == embeddings.shape[0], "Index length must match embedding count."

    # -----------------------------------------------------------------------
    # Step 4 – Compute pairwise similarity
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Compute pairwise cosine similarity ===")
    sim_matrix = cosine_similarity_matrix(embeddings)
    print(f"── Similarity matrix: {sim_matrix.shape}, "
          f"range=[{sim_matrix.min():.4f}, {sim_matrix.max():.4f}] ──\n")

    stats = compute_inter_video_stats(sim_matrix, index)
    print("── Similarity stats ──")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Step 5 – Top-k retrieval for query frames
    # -----------------------------------------------------------------------
    logger.info("=== Step 5: Top-%d retrieval for %d query frames ===",
                args.top_k, args.n_queries)

    n = len(index)
    # Spread query indices evenly across the frame list
    step = max(1, n // args.n_queries)
    query_indices = [min(i * step, n - 1) for i in range(args.n_queries)]

    query_results = batch_top_k_queries(
        query_indices, sim_matrix, index, top_k=args.top_k
    )

    for qidx, results in query_results.items():
        meta = index[qidx]
        print(
            f"  Query frame {qidx} "
            f"({meta.get('video_id','?')} @ {meta.get('timestamp','?')}s) → "
            f"top-{len(results)}: "
            + str([round(r["similarity"], 3) for r in results])
        )
    print()

    # -----------------------------------------------------------------------
    # Step 6 – Visualisations
    # -----------------------------------------------------------------------
    logger.info("=== Step 6: Generate visualisations ===")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    heatmap_path = plot_similarity_heatmap(
        sim_matrix, index,
        output_path=os.path.join(OUTPUTS_DIR, "similarity_heatmap.png"),
    )
    print(f"  Heatmap:  {heatmap_path}")

    bar_path = plot_cross_video_similarity_bar(
        stats,
        output_path=os.path.join(OUTPUTS_DIR, "cross_video_similarity_bar.png"),
    )
    print(f"  Bar chart: {bar_path}")

    for qidx, results in query_results.items():
        grid_path = plot_top_k_grid(
            qidx, results, index,
            output_path=os.path.join(OUTPUTS_DIR, f"top_k_query_{qidx}.png"),
        )
        print(f"  Top-k grid (query {qidx}): {grid_path}")

    # -----------------------------------------------------------------------
    # Step 7 – Markdown report
    # -----------------------------------------------------------------------
    logger.info("=== Step 7: Write similarity report ===")
    report_path = generate_similarity_report(
        sim_matrix, index, query_results, stats,
        output_path=os.path.join(OUTPUTS_DIR, "similarity_report.md"),
    )
    print(f"\n  Report: {report_path}")

    # -----------------------------------------------------------------------
    # Step 8 – Affective scoring (text-guided zero-shot CLIP probing)
    # -----------------------------------------------------------------------
    logger.info("=== Step 8: Compute affective axis scores ===")
    try:
        scorer = AffectiveScorer(model_name=args.model, axes=DEFAULT_AXES)
        frame_scores = scorer.score_frames(embeddings, index)

        print("\n── Affective axis scores (mean per axis) ──")
        for axis_name, scores in frame_scores.items():
            print(f"  {axis_name:12s}: {float(scores.mean()):+.4f}")
        print()

        video_scores = scorer.score_video_level(frame_scores, index)

        affective_json = scorer.save_scores(
            frame_scores, index,
            output_path=os.path.join(OUTPUTS_DIR, "affective_scores.json"),
        )
        print(f"  Affective scores JSON: {affective_json}")

        affective_heatmap = scorer.plot_heatmap(
            frame_scores, index,
            output_path=os.path.join(OUTPUTS_DIR, "affective_heatmap.png"),
        )
        print(f"  Affective heatmap:     {affective_heatmap}")

        radar_path = scorer.plot_radar(
            video_scores,
            output_path=os.path.join(OUTPUTS_DIR, "affective_radar.png"),
        )
        print(f"  Affective radar:       {radar_path}")

    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logger.warning("Affective scoring step failed (%s); continuing.", exc)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
