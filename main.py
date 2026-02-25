"""
main.py
-------
End-to-end pipeline orchestrator for the Mini GACS
Mood & Style Embedding Pipeline.

Steps
-----
1.  (Optional) Download sample videos.
2.  Extract frames from all videos found in ``data/videos/``.
    2b. Scene-adaptive extraction mode (``--scene-adaptive``).
3.  Compute CLIP embeddings for all frames.
3b. Content-adaptive deduplication — remove near-duplicate frames.
3c. Technical quality filtering — remove blurry/over-exposed/uniform frames.
4.  Calculate pairwise cosine-similarity matrix.
5.  Run top-5 retrieval for at least 3 query frames.
6.  Generate visualisations (heatmap, bar chart, retrieval grids).
7.  Write a Markdown similarity report.
8.  Affective scoring (single-prompt + multi-prompt ensemble).
9.  Vibe clustering with silhouette-based quality metrics.
10. Temporal analysis with consecutive-frame transition detection.
11. Vibe–performance regression (synthetic CTR prediction).
11b. Calibrate predictor — compute ECE and reliability diagram.
11c. Rank creatives — diversity-constrained MMR ranking with 95% CI.
12b. Text-guided creative retrieval — rank frames and videos by brand brief.
13. Embedding distribution drift detection — MMD + KS test + anomaly fraction.
15. Bayesian A/B testing on top-ranked creatives — P(A>B), Thompson sampling.
16. Occlusion saliency — spatial importance maps for affective axis scores.
14. Write structured run manifest (run_manifest.json).

Usage
-----
    # Full pipeline (downloads videos if directory is empty):
    python main.py

    # Skip download step if you already have videos:
    python main.py --skip-download

    # Use scene-adaptive frame extraction:
    python main.py --scene-adaptive

    # Override interval / model:
    python main.py --interval 2 --model openai/clip-vit-base-patch32
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np

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
from src.frame_extractor import (
    process_video_directory,
    load_metadata,
    extract_frames_scene_adaptive,
    save_metadata,
)
from src.embeddings import compute_and_save_embeddings, load_embeddings
from src.similarity import (
    cosine_similarity_matrix,
    batch_top_k_queries,
    compute_inter_video_stats,
    top_k_no_precompute,
)
from src.visualization import (
    plot_similarity_heatmap,
    plot_cross_video_similarity_bar,
    plot_top_k_grid,
    generate_similarity_report,
)
from src.affective_scoring import AffectiveScorer, DEFAULT_AXES, MULTI_PROMPT_AXES
from src.clustering import VibeClusterer, auto_n_clusters
from src.temporal_analysis import TemporalAnalyser
from src.performance_predictor import (
    generate_synthetic_performance_data,
    build_feature_names,
    VibePerformancePredictor,
)
from src.frame_deduplication import deduplicate_frames
from src.quality_filter import FrameQualityFilter
from src.experiment_manifest import PipelineManifest
from src.ranking import CreativeRanker
from src.calibration import PredictorCalibrator
from src.text_query import TextQueryRetriever
from src.drift_detector import EmbeddingDriftDetector
from src.ab_testing import CreativeABTester
from src.explainability import OcclusionSaliency
from src.pipeline_config import PipelineConfig

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
OUTPUTS_DIR = "outputs"

# N above which the full N×N similarity matrix is skipped in favour of
# top_k_no_precompute (O(N·D) per query, avoids materialising 400 MB+).
LARGE_N_MATRIX_THRESHOLD = 2000


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
    p.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        metavar="K",
        help="Number of vibe clusters (default: 0 = auto-detect via silhouette).",
    )
    p.add_argument(
        "--scene-adaptive",
        action="store_true",
        help=(
            "Use scene-adaptive frame extraction (pixel-fingerprint scene "
            "detection) instead of uniform interval sampling.  "
            "Yields one keyframe per detected scene; eliminates slow-pan "
            "frame redundancy without requiring CLIP inference."
        ),
    )
    p.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.97,
        metavar="TAU",
        help=(
            "Cosine-similarity threshold for near-duplicate frame removal "
            "(default: 0.97).  Set to 1.0 to disable deduplication."
        ),
    )
    p.add_argument(
        "--min-quality-score",
        type=float,
        default=0.25,
        metavar="Q",
        help=(
            "Minimum composite quality score [0, 1] for a frame to pass "
            "the technical quality filter (blur + exposure + information). "
            "Set to 0.0 to disable quality filtering (default: 0.25)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Initialise run manifest — collects all metrics for reproducibility
    manifest = PipelineManifest(
        config={
            "model": args.model,
            "interval_seconds": args.interval,
            "max_frames": args.max_frames,
            "scene_adaptive": args.scene_adaptive,
            "dedup_threshold": args.dedup_threshold,
            "min_quality_score": args.min_quality_score,
            "n_clusters_arg": args.n_clusters,
        }
    )

    # These are set during the embedding step when we load the CLIP model
    # directly (i.e. when --skip-embedding is not used).  They are passed to
    # TextQueryRetriever so it can reuse the already-loaded model.  When
    # --skip-embedding is used both remain None and the retriever lazy-loads.
    clip_model = None
    clip_processor = None

    # Step-result references: initialised to None so Steps 15 and 16 can
    # check availability without using fragile dir() introspection.
    ranked = None            # Set in Step 11c if creative ranking succeeds
    frame_scores = None      # Set in Step 8 if affective scoring succeeds
    affective_scorer = None  # Set in Step 8 if affective scoring succeeds

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
    elif args.scene_adaptive:
        # ---------------------------------------------------------------
        # Step 2b – Scene-adaptive extraction
        # One representative keyframe per detected visual scene, extracted
        # via pixel-level fingerprinting (no CLIP dependency).
        # ---------------------------------------------------------------
        logger.info("=== Step 2 (scene-adaptive): Extract scene keyframes ===")
        import glob as _glob
        extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        video_files = sorted([
            f for f in (
                os.path.join(VIDEOS_DIR, fn)
                for fn in os.listdir(VIDEOS_DIR)
                if os.path.isfile(os.path.join(VIDEOS_DIR, fn))
            )
            if os.path.splitext(f)[1].lower() in extensions
        ])
        metadata = []
        os.makedirs(FRAMES_DIR, exist_ok=True)
        os.makedirs(METADATA_DIR, exist_ok=True)
        for vpath in video_files:
            from pathlib import Path as _Path
            vid_id = _Path(vpath).stem
            vid_frames_dir = os.path.join(FRAMES_DIR, vid_id)
            try:
                meta = extract_frames_scene_adaptive(
                    vpath,
                    vid_frames_dir,
                    coarse_fps=2.0,
                    transition_threshold=0.20,
                    max_scenes=args.max_frames,
                )
                csv_out = os.path.join(METADATA_DIR, f"{vid_id}_metadata.csv")
                save_metadata(meta, csv_out)
                metadata.extend(meta)
                logger.info(
                    "Scene-adaptive: %d keyframes from '%s'.", len(meta), vid_id
                )
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error("Skipping '%s': %s", vpath, exc)
        if metadata:
            save_metadata(metadata, combined_csv)
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

    manifest.record(
        "embeddings",
        shape=list(embeddings.shape),
        dtype=str(embeddings.dtype),
        model=args.model,
        n_frames_in=len(metadata),
        n_frames_embedded=len(index),
    )

    # -----------------------------------------------------------------------
    # Step 3b – Content-adaptive frame deduplication
    #
    # Remove near-duplicate frames caused by slow pans, static shots, or
    # fades.  Without this step the similarity matrix shows block-diagonal
    # artifacts driven by temporal proximity rather than style, cluster sizes
    # are biased toward slower-paced videos, and temporal coherence scores
    # are artificially inflated.
    # -----------------------------------------------------------------------
    logger.info("=== Step 3b: Content-adaptive frame deduplication (tau=%.3f) ===",
                args.dedup_threshold)
    if args.dedup_threshold < 1.0:
        embeddings, index, kept_mask = deduplicate_frames(
            embeddings, index, similarity_threshold=args.dedup_threshold
        )
        print(
            f"\n── Deduplication: kept {len(index)} / {kept_mask.shape[0]} frames "
            f"({100.0 * len(index) / max(kept_mask.shape[0], 1):.1f}%) ──\n"
        )
        manifest.record(
            "deduplication",
            threshold=args.dedup_threshold,
            n_before=int(kept_mask.shape[0]),
            n_after=len(index),
            reduction_pct=round(
                100.0 * (kept_mask.shape[0] - len(index)) / max(kept_mask.shape[0], 1), 2
            ),
        )
    else:
        logger.info("Deduplication skipped (--dedup-threshold=1.0).")

    # -----------------------------------------------------------------------
    # Step 3c – Technical quality filtering
    #
    # Remove frames that are motion-blurred, over/under-exposed, or near-
    # uniform (fade-to-black/white).  Such frames:
    #   • Bias video-level affective scores (blurry frames score lower on
    #     "luxury" and "complexity" even for inherently high-scoring content).
    #   • Create spurious "blown-out" or "fade" clusters unrelated to style.
    #   • Trigger false scene transitions in temporal analysis.
    # This step runs on CPU with PIL/NumPy — negligible vs CLIP inference.
    # -----------------------------------------------------------------------
    logger.info("=== Step 3c: Technical quality filtering (min_score=%.2f) ===",
                args.min_quality_score)
    if args.min_quality_score > 0.0:
        quality_filter = FrameQualityFilter(
            min_composite_score=args.min_quality_score
        )
        quality_scores = quality_filter.score_frames(index)
        n_before_qf = len(index)
        index, qf_mask = quality_filter.filter_frames(index, quality_scores)
        # Keep embeddings aligned with filtered index
        embeddings = embeddings[qf_mask]
        n_after_qf = len(index)
        print(
            f"\n── Quality filter: kept {n_after_qf} / {n_before_qf} frames "
            f"({100.0 * n_after_qf / max(n_before_qf, 1):.1f}%) ──\n"
        )
        manifest.record(
            "quality_filter",
            min_composite_score=args.min_quality_score,
            n_before=n_before_qf,
            n_after=n_after_qf,
            reduction_pct=round(
                100.0 * (n_before_qf - n_after_qf) / max(n_before_qf, 1), 2
            ),
        )
    else:
        logger.info("Quality filtering skipped (--min-quality-score=0.0).")

    # -----------------------------------------------------------------------
    # Step 4 – Compute pairwise similarity
    #
    # For N ≤ 2000 frames we precompute the full N×N matrix (16 MB).
    # For larger datasets we use top_k_no_precompute which materialises only
    # one O(N) row per query instead of the full O(N²) matrix.
    # -----------------------------------------------------------------------
    _LARGE_N = embeddings.shape[0]

    if _LARGE_N <= LARGE_N_MATRIX_THRESHOLD:
        sim_matrix = cosine_similarity_matrix(embeddings)
        print(f"── Similarity matrix: {sim_matrix.shape}, "
              f"range=[{sim_matrix.min():.4f}, {sim_matrix.max():.4f}] ──\n")
    else:
        # Large dataset path — defer full matrix; heatmap/vis steps are skipped
        logger.warning(
            "N=%d > %d: skipping full N×N matrix. "
            "Retrieval will use top_k_no_precompute; heatmap skipped.",
            _LARGE_N, LARGE_N_MATRIX_THRESHOLD,
        )
        sim_matrix = None

    if sim_matrix is not None:
        stats = compute_inter_video_stats(sim_matrix, index)
        print("── Similarity stats ──")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        print()
        manifest.record("similarity", **stats, n_frames=len(index))
    else:
        stats = {}
        logger.info("Similarity stats skipped (large-N path).")

    # -----------------------------------------------------------------------
    # Step 5 – Top-k retrieval for query frames
    # -----------------------------------------------------------------------
    logger.info("=== Step 5: Top-%d retrieval for %d query frames ===",
                args.top_k, args.n_queries)

    n = len(index)
    # Spread query indices evenly across the frame list
    step = max(1, n // args.n_queries)
    query_indices = [min(i * step, n - 1) for i in range(args.n_queries)]

    if sim_matrix is not None:
        # Fast path: index into precomputed matrix row
        query_results = batch_top_k_queries(
            query_indices, sim_matrix, index, top_k=args.top_k
        )
    else:
        # Memory-efficient path: O(N·D) per query, no N×N matrix
        query_results = top_k_no_precompute(
            query_indices, embeddings, index, top_k=args.top_k
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

    if sim_matrix is not None:
        heatmap_path = plot_similarity_heatmap(
            sim_matrix, index,
            output_path=os.path.join(OUTPUTS_DIR, "similarity_heatmap.png"),
        )
        print(f"  Heatmap:  {heatmap_path}")

    if stats:
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
    if sim_matrix is not None:
        report_path = generate_similarity_report(
            sim_matrix, index, query_results, stats,
            output_path=os.path.join(OUTPUTS_DIR, "similarity_report.md"),
        )
        print(f"\n  Report: {report_path}")

    # -----------------------------------------------------------------------
    # Step 8 – Affective scoring (text-guided zero-shot CLIP probing)
    # -----------------------------------------------------------------------
    logger.info("=== Step 8: Compute affective axis scores ===")
    frame_scores: Optional[dict] = None  # None = step failed; {} = no axes configured
    try:
        scorer = AffectiveScorer(model_name=args.model, axes=DEFAULT_AXES)
        frame_scores = scorer.score_frames(embeddings, index)
        print("\n── Affective axis scores (mean per axis) ──")
        for axis_name, scores in frame_scores.items():
            print(f"  {axis_name:12s}: {float(scores.mean()):+.4f}")
        print()

        # Multi-prompt ensemble scoring (more reliable than single-prompt)
        print("── Ensemble affective scores (5 prompts/pole, with confidence) ──")
        ens_scores, ens_confidence = scorer.score_frames_ensemble(
            embeddings, multi_prompt_axes=MULTI_PROMPT_AXES, index=index
        )
        for axis_name, scores in ens_scores.items():
            conf = ens_confidence[axis_name].mean()
            print(f"  {axis_name:12s}: {float(scores.mean()):+.4f}  "
                  f"(confidence std={conf:.4f})")
        print()

        video_scores = scorer.score_video_level(ens_scores, index)

        affective_json = scorer.save_scores(
            ens_scores, index,
            output_path=os.path.join(OUTPUTS_DIR, "affective_scores.json"),
        )
        print(f"  Affective scores JSON: {affective_json}")

        affective_heatmap = scorer.plot_heatmap(
            ens_scores, index,
            output_path=os.path.join(OUTPUTS_DIR, "affective_heatmap.png"),
        )
        print(f"  Affective heatmap:     {affective_heatmap}")

        radar_path = scorer.plot_radar(
            video_scores,
            output_path=os.path.join(OUTPUTS_DIR, "affective_radar.png"),
        )
        print(f"  Affective radar:       {radar_path}")

        manifest.record(
            "affective_scoring",
            axes=list(ens_scores.keys()),
            mean_confidence={k: round(float(v.mean()), 4) for k, v in ens_confidence.items()},
        )
        manifest.add_artifact(affective_json, "Frame-level ensemble affective scores")
        manifest.add_artifact(affective_heatmap, "Affective score heatmap (axes x frames)")
        manifest.add_artifact(radar_path, "Affective radar chart (per-video profiles)")

        # Gap-corrected scoring (Liang et al. NeurIPS 2022 modality-gap fix)
        # compares absolute axis magnitudes to the standard ensemble scores.
        try:
            gap_scores = scorer.score_frames_gap_corrected(embeddings, index)
            print("── Gap-corrected scores (absolute magnitudes after centering) ──")
            for axis_name, gc_scores in gap_scores.items():
                raw_mean = float(ens_scores.get(axis_name, gc_scores).mean())
                gc_mean  = float(gc_scores.mean())
                print(f"  {axis_name:12s}: raw={raw_mean:+.4f}  "
                      f"gap-corrected={gc_mean:+.4f}  "
                      f"Δ={gc_mean - raw_mean:+.4f}")
            print()
            manifest.record(
                "modality_gap_correction",
                axes=list(gap_scores.keys()),
                mean_delta={
                    k: round(
                        float(gap_scores[k].mean())
                        - float(ens_scores.get(k, gap_scores[k]).mean()),
                        4,
                    )
                    for k in gap_scores
                },
            )
        except Exception as _gap_exc:  # noqa: BLE001
            logger.debug("Gap correction diagnostic failed: %s", _gap_exc)

        # Update frame_scores to use ensemble for downstream modules
        frame_scores = ens_scores
        affective_scorer = scorer  # expose for Step 16 (occlusion saliency)

    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logger.warning("Affective scoring step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 9 – Vibe clustering
    # -----------------------------------------------------------------------
    logger.info("=== Step 9: Vibe clustering ===")
    try:
        n_frames = embeddings.shape[0]
        k = args.n_clusters if args.n_clusters >= 2 else auto_n_clusters(
            embeddings, max_k=min(10, n_frames - 1)
        )
        k = max(2, min(k, n_frames - 1))  # clamp to valid range

        clusterer = VibeClusterer(n_clusters=k)
        labels = clusterer.fit(embeddings)

        # Cluster quality metrics (silhouette + Davies-Bouldin)
        quality = clusterer.cluster_quality(embeddings, labels)
        print(f"\n── Vibe clustering: {k} clusters, {n_frames} frames ──")
        print(f"  Silhouette score:   {quality['silhouette']:.4f}  "
              f"(higher better, >0.5 = well-separated)")
        print(f"  Davies-Bouldin idx: {quality['davies_bouldin']:.4f}  "
              f"(lower better, <1.0 = compact clusters)")
        summary = clusterer.cluster_summary(labels, index, embeddings=embeddings)
        for cid, info in summary.items():
            print(f"  Cluster {cid}: {info['size']} frames  "
                  f"videos={info['video_distribution']}")
        print()

        scatter_path = clusterer.plot_scatter(
            embeddings, labels, index,
            output_path=os.path.join(OUTPUTS_DIR, "vibe_cluster_scatter.png"),
            projection="pca",
            title=f"Vibe Cluster Map – {k} clusters (PCA)",
        )
        print(f"  Cluster scatter: {scatter_path}")

        assign_path = clusterer.save_cluster_assignments(
            labels, index,
            output_path=os.path.join(OUTPUTS_DIR, "cluster_assignments.json"),
        )
        print(f"  Cluster assignments: {assign_path}")

        manifest.record(
            "clustering",
            n_clusters=k,
            silhouette=round(quality["silhouette"], 4),
            davies_bouldin=round(quality["davies_bouldin"], 4),
            cluster_sizes={str(cid): info["size"] for cid, info in summary.items()},
        )
        manifest.add_artifact(scatter_path, "2-D PCA vibe cluster scatter plot")

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Clustering step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 10 – Temporal analysis (narrative arc)
    # -----------------------------------------------------------------------
    logger.info("=== Step 10: Temporal analysis ===")
    try:
        ta = TemporalAnalyser(window=3)
        temporal_curve = ta.compute_temporal_curve(embeddings, index)

        # Use consecutive-frame similarities for scene-cut detection
        # (more accurate than detecting drops in the windowed curve).
        consec_sims = ta.compute_consecutive_similarities(embeddings, index)
        transitions = ta.detect_scene_transitions(consec_sims, threshold=0.12)
        pacing = ta.pacing_score(temporal_curve)
        coherence = ta.coherence_score(temporal_curve)
        pacing_rate = ta.pacing_rate_per_second(transitions, index)

        print(f"\n── Temporal analysis ──")
        print(f"  Coherence score:     {coherence:+.4f}  (higher = smoother narrative)")
        print(f"  Pacing score:        {pacing:.6f}  (variance of windowed curve)")
        print(f"  Pacing rate:         {pacing_rate:.4f} cuts/second")
        print(f"  Scene transitions:   {len(transitions)} detected at frames {transitions[:10]}")
        print()

        per_vid_stats = ta.per_video_stats(temporal_curve, index)
        for vid, vstats in per_vid_stats.items():
            print(f"  {vid}: coherence={vstats['coherence']:.3f}, "
                  f"pacing={vstats['pacing']:.5f}, "
                  f"transitions={int(vstats['n_transitions'])}")
        print()

        arc_path = ta.plot_narrative_arc(
            temporal_curve, transitions, index,
            output_path=os.path.join(OUTPUTS_DIR, "narrative_arc.png"),
        )
        print(f"  Narrative arc: {arc_path}")

        pacing_path = ta.plot_pacing_comparison(
            per_vid_stats,
            output_path=os.path.join(OUTPUTS_DIR, "pacing_comparison.png"),
        )
        print(f"  Pacing chart:  {pacing_path}")

        temporal_json = ta.save_temporal_stats(
            temporal_curve, index,
            output_path=os.path.join(OUTPUTS_DIR, "temporal_stats.json"),
        )
        print(f"  Temporal JSON: {temporal_json}")

        manifest.record(
            "temporal_analysis",
            overall_coherence=round(coherence, 4),
            overall_pacing=round(pacing, 6),
            pacing_rate_per_second=round(pacing_rate, 4),
            n_transitions=len(transitions),
            per_video={vid: {k: round(v, 4) for k, v in s.items()}
                       for vid, s in per_vid_stats.items()},
        )
        manifest.add_artifact(arc_path, "Narrative arc – temporal similarity curve")

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Temporal analysis step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 11 – Vibe–Performance Regression (synthetic CTR prediction)
    # -----------------------------------------------------------------------
    logger.info("=== Step 11: Vibe–performance regression ===")
    try:
        if frame_scores is None:
            logger.warning("No affective scores available (step 8 failed); skipping predictor.")
        else:
            features, ctr_labels = generate_synthetic_performance_data(
                embeddings, frame_scores, index, target="ctr"
            )
            feat_names = build_feature_names(frame_scores)

            predictor = VibePerformancePredictor(model_type="ridge")
            cv_results = predictor.cross_validate(features, ctr_labels, n_splits=5)

            print(f"\n── Vibe → CTR predictor (ridge, 5-fold CV) ──")
            print(f"  Spearman ρ:  {cv_results['mean_spearman']:.4f} "
                  f"± {cv_results['std_spearman']:.4f}")
            print(f"  RMSE:        {cv_results['mean_rmse']:.4f} "
                  f"± {cv_results['std_rmse']:.4f}")

            predictor.fit(features, ctr_labels, feature_names=feat_names)
            importance = predictor.feature_importance(top_k=6)
            print("\n  Top feature importances (vibe → CTR):")
            for fname, score in importance:
                print(f"    {fname:20s}: {score:.4f}")
            print()

            cv_chart = predictor.plot_cv_results(
                cv_results,
                output_path=os.path.join(OUTPUTS_DIR, "predictor_cv_results.png"),
                title="CTR Predictor — Spearman ρ per Fold",
            )
            print(f"  CV chart:          {cv_chart}")

            imp_chart = predictor.plot_feature_importance(
                output_path=os.path.join(OUTPUTS_DIR, "predictor_feature_importance.png"),
                feature_names=feat_names,
            )
            print(f"  Importance chart:  {imp_chart}")

            predicted_ctr = predictor.predict(features)
            scatter_chart = predictor.plot_predicted_vs_actual(
                ctr_labels, predicted_ctr,
                output_path=os.path.join(OUTPUTS_DIR, "predictor_predicted_vs_actual.png"),
            )
            print(f"  Pred vs actual:    {scatter_chart}")

            model_json = predictor.save_model(
                output_path=os.path.join(OUTPUTS_DIR, "predictor_model.json"),
            )
            print(f"  Model JSON:        {model_json}")

            manifest.record(
                "performance_predictor",
                model_type="ridge",
                mean_spearman=round(cv_results["mean_spearman"], 4),
                std_spearman=round(cv_results["std_spearman"], 4),
                mean_rmse=round(cv_results["mean_rmse"], 4),
                n_features=features.shape[1],
                top_features={fname: round(score, 4) for fname, score in importance},
            )

            # -------------------------------------------------------------------
            # Step 11b – Calibrate predictor (Platt scaling + reliability diagram)
            #
            # The ridge predictor outputs raw regression scores in [0, 1] that
            # are NOT calibrated probabilities.  Post-hoc Platt scaling fits a
            # logistic sigmoid on the cross-validated OOF predictions, so the
            # output scores have a proper frequentist interpretation:
            # score = 0.7 means ~70% of creatives with that score land above
            # median CTR.  ECE < 0.05 is the production-grade target.
            # -------------------------------------------------------------------
            logger.info("=== Step 11b: Calibrate predictor ===")
            try:
                calibrator = PredictorCalibrator(method="platt")
                # Use the point estimates on training data as a proxy for
                # OOF predictions (honest calibration requires a held-out set;
                # this is a prototype-grade approximation for demo purposes).
                # TODO(production): replace `predicted_ctr` with OOF predictions
                # collected from cross_validate() to avoid in-sample over-confidence.
                logger.warning(
                    "Calibrator fitted on in-sample predictions — "
                    "use out-of-fold (OOF) predictions from cross_validate() "
                    "for honest calibration in production."
                )
                calibrator.fit(predicted_ctr, ctr_labels)
                calibrated_ctr = calibrator.transform(predicted_ctr)
                ece = calibrator.expected_calibration_error(calibrated_ctr, ctr_labels)

                print(f"\n── Predictor calibration (Platt scaling) ──")
                print(f"  ECE (uncalibrated proxy): {ece:.4f}  (< 0.05 = well-calibrated)")

                cal_diagram = calibrator.plot_reliability_diagram(
                    calibrated_ctr, ctr_labels,
                    output_path=os.path.join(OUTPUTS_DIR, "calibration_reliability.png"),
                    raw_scores=predicted_ctr,
                    title="CTR Predictor Calibration — Reliability Diagram",
                )
                print(f"  Reliability diagram: {cal_diagram}")

                cal_params = calibrator.save_calibration_params(
                    output_path=os.path.join(OUTPUTS_DIR, "calibration_params.json"),
                )
                print(f"  Calibration params:  {cal_params}")

                manifest.record(
                    "calibration",
                    method="platt",
                    ece=round(ece, 4),
                )
                manifest.add_artifact(cal_diagram, "Reliability diagram — calibrated CTR")

            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Calibration step failed (%s); continuing.", exc)
                calibrated_ctr = predicted_ctr  # fall back to raw scores

            # -------------------------------------------------------------------
            # Step 11c – Rank creatives (MMR diversity + bootstrap CI)
            #
            # Pure score ranking surfaces the N most similar top-scoring frames,
            # not the N most useful.  CreativeRanker applies MMR re-ranking:
            # each successive selection maximises the trade-off between predicted
            # CTR (or its CI lower bound for conservative ranking) and novelty
            # relative to already-selected items.  The result is the top-10
            # most confidently high-performing AND visually diverse creatives.
            # -------------------------------------------------------------------
            logger.info("=== Step 11c: Rank creatives (MMR + bootstrap CI) ===")
            try:
                ranker = CreativeRanker(
                    predictor,
                    lambda_mmr=0.6,
                    n_bootstrap=200,
                    ci_level=0.95,
                )
                ranked = ranker.rank(
                    embeddings, features, index,
                    top_k=min(10, len(index)),
                    use_ci_lower=True,
                )

                print(f"\n── Top-{len(ranked)} creatives (MMR-ranked, 95% CI) ──")
                for rc in ranked[:5]:
                    print(
                        f"  #{rc.rank:2d}  {rc.video_id} @ {rc.timestamp:.1f}s  "
                        f"score={rc.predicted_score:.3f}  "
                        f"CI=[{rc.ci_lower:.3f}, {rc.ci_upper:.3f}]  "
                        f"diversity={rc.diversity_score:.3f}"
                    )
                if len(ranked) > 5:
                    print(f"  ... ({len(ranked) - 5} more)")
                print()

                ranking_chart = ranker.plot_ranking(
                    ranked,
                    output_path=os.path.join(OUTPUTS_DIR, "creative_ranking.png"),
                    title="Creative Ranking — Predicted CTR with 95% Bootstrap CI",
                )
                print(f"  Ranking chart:  {ranking_chart}")

                ranking_json = ranker.save_ranking(
                    ranked,
                    output_path=os.path.join(OUTPUTS_DIR, "creative_ranking.json"),
                )
                print(f"  Ranking JSON:   {ranking_json}")

                manifest.record(
                    "creative_ranking",
                    lambda_mmr=0.6,
                    n_bootstrap=200,
                    top_k=len(ranked),
                    top_video=ranked[0].video_id if ranked else None,
                    top_score=round(ranked[0].predicted_score, 4) if ranked else None,
                )
                manifest.add_artifact(ranking_chart, "MMR-ranked creative bar chart")

            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Creative ranking step failed (%s); continuing.", exc)

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Performance regression step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 12b – Text-guided creative retrieval
    # -----------------------------------------------------------------------
    logger.info("Step 12b – Text-guided creative retrieval.")
    print("\n── Step 12b: Text-Guided Creative Retrieval ──")
    try:
        # clip_model and clip_processor are set at the top of main() and
        # populated during the embedding step when the CLIP model is loaded.
        # When --skip-embedding is used both remain None and the retriever
        # lazy-loads its own model instance.
        retriever = TextQueryRetriever(
            model=clip_model,
            processor=clip_processor,
        )

        # Representative sample briefs for demonstration
        _DEMO_BRIEFS = {
            "luxury_warmth": "warm, golden, opulent luxury aesthetic — premium feel",
            "energy_action": "high energy, dynamic, fast-paced, exciting movement",
            "calm_minimal":  "serene, minimalist, clean, contemplative — low key",
        }

        for brief_name, brief_text in _DEMO_BRIEFS.items():
            results = retriever.query(brief_text, embeddings, index, top_k=5)
            chart_path = retriever.plot_query_results(
                results,
                output_path=os.path.join(OUTPUTS_DIR, f"text_query_{brief_name}.png"),
                title=f"Brief: {brief_name}",
            )
            retriever.save_results(
                results,
                output_path=os.path.join(OUTPUTS_DIR, f"text_query_{brief_name}.json"),
            )
            print(f"  [{brief_name}] top match: {results[0].video_id} "
                  f"@ {results[0].timestamp:.1f}s  sim={results[0].similarity:.4f}")
            manifest.add_artifact(chart_path, f"Text query chart: {brief_name}")

        # Multi-brief comparison heatmap
        comparison = retriever.multi_brief_comparison(
            _DEMO_BRIEFS, embeddings, index
        )
        heatmap_path = retriever.plot_brief_comparison_heatmap(
            comparison,
            output_path=os.path.join(OUTPUTS_DIR, "brief_alignment_heatmap.png"),
            title="Brand Brief × Video Alignment",
        )
        print(f"  Brief alignment heatmap: {heatmap_path}")
        manifest.add_artifact(heatmap_path, "Brand brief × video alignment heatmap")

        # Video ranking by primary brief
        primary_brief = _DEMO_BRIEFS["luxury_warmth"]
        video_ranking = retriever.rank_videos_by_brief(primary_brief, embeddings, index)
        print(f"  Video ranking by '{primary_brief[:40]}…':")
        for row in video_ranking:
            print(f"    {row['video_id']:15s}  score={row['score']:.4f}  "
                  f"({row['n_frames']} frames)")

        manifest.record(
            "text_query",
            n_briefs=len(_DEMO_BRIEFS),
            primary_brief=primary_brief[:60],
            top_video=video_ranking[0]["video_id"] if video_ranking else None,
        )

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Text-guided retrieval step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 13 – Embedding distribution drift detection
    # -----------------------------------------------------------------------
    logger.info("Step 13 – Embedding distribution drift detection.")
    print("\n── Step 13: Embedding Distribution Drift Detection ──")
    try:
        # Demo: compare embeddings from the first video (reference) to
        # embeddings from all other videos (new batch).  In production,
        # the reference would be the batch used to train the predictor.
        video_ids_arr = np.array([e.get("video_id", "") for e in index])
        unique_vids = list(dict.fromkeys(video_ids_arr))  # ordered unique

        if len(unique_vids) >= 2:
            ref_mask = video_ids_arr == unique_vids[0]
            new_mask = ~ref_mask
            ref_embs = embeddings[ref_mask]
            new_embs = embeddings[new_mask]

            if ref_embs.shape[0] >= 5 and new_embs.shape[0] >= 5:
                detector = EmbeddingDriftDetector(n_components=min(10, ref_embs.shape[0] - 1))
                detector.fit(ref_embs)
                drift_report = detector.detect(new_embs)

                print(f"  Reference video:  {unique_vids[0]}  ({ref_embs.shape[0]} frames)")
                print(f"  New-batch videos: {unique_vids[1:]}  ({new_embs.shape[0]} frames)")
                print(f"  MMD:              {drift_report.mmd:.4f}")
                print(f"  KS p-val (min):   {drift_report.ks_pvalue_min:.4f}")
                print(f"  Distribution drifted: {drift_report.is_drifted}")
                print(f"  Anomaly fraction:     {drift_report.anomaly_fraction:.3f}")

                if drift_report.is_drifted:
                    logger.warning(
                        "Drift detected (KS min p=%.4f < %.2f): "
                        "new batch may be out-of-distribution for the predictor.",
                        drift_report.ks_pvalue_min, detector.alpha,
                    )

                pca_plot = detector.plot_pca_comparison(
                    new_embs,
                    output_path=os.path.join(OUTPUTS_DIR, "drift_pca.png"),
                )
                print(f"  PCA comparison:   {pca_plot}")
                manifest.add_artifact(pca_plot, "Embedding drift PCA scatter (ref vs new)")

                drift_json = detector.save_report(
                    drift_report,
                    output_path=os.path.join(OUTPUTS_DIR, "drift_report.json"),
                )
                print(f"  Drift report:     {drift_json}")
                manifest.add_artifact(drift_json, "Embedding drift report JSON")

                manifest.record(
                    "drift_detection",
                    mmd=round(drift_report.mmd, 4),
                    ks_pvalue_min=round(drift_report.ks_pvalue_min, 4),
                    is_drifted=drift_report.is_drifted,
                    anomaly_fraction=round(drift_report.anomaly_fraction, 3),
                    n_reference=drift_report.n_reference,
                    n_new=drift_report.n_new,
                )
            else:
                logger.info(
                    "Too few frames in reference (%d) or new batch (%d) for drift analysis.",
                    ref_embs.shape[0], new_embs.shape[0],
                )
                print("  Skipped: not enough frames per group for drift analysis.")
        else:
            print("  Skipped: only one video present; drift requires at least 2.")

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Drift detection step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 15 – Bayesian A/B testing on the top-ranked creatives
    #
    # Pure ranking by point estimate doesn't answer "are we statistically
    # confident that creative #1 outperforms creative #2?"  CreativeABTester
    # computes P(A > B) under the Gaussian CI model from CreativeRanker and
    # validates it with Thompson sampling.  The tournament heatmap gives a
    # holistic view of pairwise dominance across the whole top-k set.
    # -----------------------------------------------------------------------
    logger.info("=== Step 15: Bayesian A/B testing on top-ranked creatives ===")
    print("\n── Step 15: Bayesian A/B Testing ──")
    try:
        # `ranked` is initialised to None at the start of main(); it is
        # populated by Step 11c when creative ranking succeeds.
        if ranked is not None and len(ranked) >= 2:
            ab_tester = CreativeABTester(win_threshold=0.80, n_thompson=10_000)

            # Pairwise test: #1 vs #2 (most actionable comparison)
            ab_result = ab_tester.compare_pair(ranked[0], ranked[1])
            print(
                f"  #{ranked[0].rank} vs #{ranked[1].rank}: "
                f"P(A>B)={ab_result.win_probability:.3f}  "
                f"d={ab_result.cohens_d:+.2f}  "
                f"→ {ab_result.recommendation}"
            )

            ab_chart = ab_tester.plot_comparison(
                ab_result, ranked[0], ranked[1],
                output_path=os.path.join(OUTPUTS_DIR, "ab_comparison.png"),
                title="Bayesian A/B: #1 vs #2 Creative",
            )
            print(f"  A/B chart:         {ab_chart}")

            # All-pairs comparison for top-5 creatives
            top5 = ranked[:min(5, len(ranked))]
            all_ab = ab_tester.compare_all_pairs(top5)

            decisive = [r for r in all_ab if r.recommendation != "inconclusive"]
            print(
                f"  All-pairs (top-5): {len(all_ab)} comparisons, "
                f"{len(decisive)} decisive (P > 0.80 or < 0.20)"
            )

            ab_json = ab_tester.save_results(
                all_ab,
                output_path=os.path.join(OUTPUTS_DIR, "ab_results.json"),
            )
            print(f"  A/B JSON:          {ab_json}")

            tournament = ab_tester.plot_tournament_heatmap(
                top5,
                output_path=os.path.join(OUTPUTS_DIR, "ab_tournament.png"),
                title="P(row > col) — Top-5 Creative Tournament",
            )
            print(f"  Tournament:        {tournament}")

            # Thompson sampling: select top-3 from top-5 by win rate
            thompson_top3 = ab_tester.thompson_select(top5, n_select=3)
            print(
                f"  Thompson top-3 (from top-5): "
                + str([f"#{top5[i].rank}" for i in thompson_top3])
            )

            manifest.record(
                "ab_testing",
                n_pairs=len(all_ab),
                n_decisive=len(decisive),
                top1_vs_top2_win_prob=round(ab_result.win_probability, 4),
                recommendation=ab_result.recommendation,
                thompson_top3=[int(top5[i].rank) for i in thompson_top3],
            )
            manifest.add_artifact(tournament, "A/B tournament heatmap")
        else:
            print("  Skipped: no ranked creatives available (Step 11c did not run).")

    except (RuntimeError, ValueError, OSError, NameError) as exc:
        logger.warning("A/B testing step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 16 – Occlusion saliency for representative frames
    #
    # For each query frame from Step 5, compute which spatial regions of the
    # image most contributed to each affective axis score.  This turns the
    # black-box "energy = 0.72" into an interpretable spatial map — a
    # prerequisite for actionable creative recommendations ("the top-left
    # product shot is what drives your luxury score").
    #
    # Only runs when affective scores are available AND frame files exist on
    # disk (not available in --skip-extraction mode on CI).
    # -----------------------------------------------------------------------
    logger.info("=== Step 16: Occlusion saliency for representative frames ===")
    print("\n── Step 16: Occlusion Saliency ──")
    try:
        # frame_scores and affective_scorer are initialised to None at the
        # start of main(); both are set by Step 8 when affective scoring
        # succeeds.
        if (
            frame_scores is not None
            and len(frame_scores) > 0
            and affective_scorer is not None
            and len(index) > 0
        ):
            occ = OcclusionSaliency(affective_scorer, grid_rows=4, grid_cols=4)

            # Pick up to 3 query frames that have image files on disk
            saliency_frames = [
                (i, e) for i, e in enumerate(index)
                if os.path.exists(e.get("file_path", ""))
            ][:3]

            if saliency_frames:
                for global_idx, meta in saliency_frames:
                    baseline = {
                        ax: float(frame_scores[ax][global_idx])
                        for ax in frame_scores
                        if global_idx < len(frame_scores[ax])
                    }
                    saliency = occ.compute_saliency(
                        meta["file_path"], baseline
                    )
                    out_name = (
                        f"saliency_{meta.get('video_id','v')}"
                        f"_f{meta.get('frame_idx', global_idx)}.png"
                    )
                    sal_path = occ.plot_saliency_overlay(
                        meta["file_path"], saliency,
                        output_path=os.path.join(OUTPUTS_DIR, out_name),
                    )
                    # Show top-3 patches for the energy axis as example
                    if "energy" in saliency:
                        top3 = occ.top_important_patches(saliency, "energy", top_k=3)
                        print(
                            f"  {meta.get('video_id')} frame {meta.get('frame_idx')}: "
                            f"top energy patches (row,col,imp): "
                            + str([(r, c, round(imp, 3)) for r, c, imp in top3])
                        )
                    print(f"  Saliency overlay:  {sal_path}")
                    manifest.add_artifact(sal_path, f"Occlusion saliency: {out_name}")

                manifest.record("occlusion_saliency", n_frames=len(saliency_frames))
            else:
                print("  Skipped: no frame files found on disk.")
        else:
            print(
                "  Skipped: affective scores or scorer not available "
                "(run without --skip-embedding)."
            )

    except (RuntimeError, ValueError, OSError, NameError) as exc:
        logger.warning("Occlusion saliency step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 17 – Online learning demo
    # Warm-starts an OnlinePredictor from the batch-trained Ridge model,
    # ingest a small stream of simulated incoming performance data, checks
    # for concept drift, and logs predictive-interval statistics.
    # -----------------------------------------------------------------------
    logger.info("=== Step 17: Online learning demo ===")
    try:
        from src.online_updater import OnlinePredictor
        from src.performance_predictor import generate_synthetic_performance_data

        # Build a quick batch predictor to warm-start from
        _tmp_predictor = VibePerformancePredictor(n_epochs=50, random_state=42)
        _feat_names = build_feature_names()
        _syn_X, _syn_y = generate_synthetic_performance_data(
            n_samples=60, noise_std=0.05, random_state=42
        )
        _tmp_predictor.fit(_syn_X, _syn_y, feature_names=_feat_names)

        online = OnlinePredictor.from_batch_predictor(
            _tmp_predictor, max_window=200
        )

        # Simulate arriving performance stream in 5 mini-batches
        rng_stream = np.random.default_rng(7)
        for batch_i in range(5):
            X_new, y_new = generate_synthetic_performance_data(
                n_samples=8, noise_std=0.05, random_state=int(rng_stream.integers(1000))
            )
            online.partial_fit(X_new, y_new)

        res = online.residual_stats()
        print(
            f"\n── Online predictor after 40 new samples ──\n"
            f"  Window size:      {online.window_size}\n"
            f"  Mean residual:    {res.get('mean_residual', 0.0):+.4f}\n"
            f"  Std residual:     {res.get('std_residual', 0.0):.4f}\n"
            f"  Drift alarm:      {online.drift_detected()}"
        )

        # Inject 20 highly biased samples to trigger CUSUM alarm
        X_drift = rng_stream.random((20, online.n_features)).astype(np.float32)
        y_drift = np.full(20, 5.0)   # artificially high labels → large residuals
        online.partial_fit(X_drift, y_drift)
        ds = online.drift_summary()
        print(
            f"  After drift injection: alarm={ds['drift_alarm']}  "
            f"cusum_pos={ds['cusum_pos']:.2f}  "
            f"cusum_neg={ds['cusum_neg']:.2f}"
        )

        online_path = os.path.join(OUTPUTS_DIR, "online_predictor.npz")
        online.save(online_path)
        print(f"  Online predictor saved: {online_path}")
        manifest.record(
            "online_learning",
            window_size=online.window_size,
            n_updates=online.n_updates,
            drift_alarm=online.drift_detected(),
        )
        manifest.add_artifact(online_path, "OnlinePredictor model state")

    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        logger.warning("Online learning step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 18 – Audio feature extraction and cross-modal discord scoring
    # -----------------------------------------------------------------------
    print("\n── Step 18: Audio features & cross-modal discord ──────────────────")
    try:
        from src.audio_features import (
            score_video_audio,
            generate_synthetic_audio_features,
            map_to_affective_axes,
            cross_modal_discord_score,
            AudioExtractionError,
        )

        video_paths = sorted(
            {entry.get("video_path", entry.get("file_path", "")) for entry in index
             if entry.get("video_path") or entry.get("file_path", "").endswith(".mp4")}
        )
        # Deduplicate to unique video files
        video_paths = [p for p in video_paths if os.path.isfile(p) and p.endswith(".mp4")]

        if video_paths and video_level_scores:
            print(f"  Attempting audio extraction from {len(video_paths)} video(s)…")
            audio_results: Dict[str, Optional[Dict[str, float]]] = {}
            for vp in video_paths[:3]:  # cap at 3 to keep demo quick
                vid_id = os.path.splitext(os.path.basename(vp))[0]
                audio_aff = score_video_audio(vp)
                audio_results[vid_id] = audio_aff

            # Fall back to synthetic features when ffmpeg unavailable
            if all(v is None for v in audio_results.values()):
                logger.info("ffmpeg unavailable — using synthetic audio features for demo.")
                synth = generate_synthetic_audio_features(
                    len(audio_results) or 3, seed=config.seed if hasattr(config, "seed") else 42
                )
                for i, vid_id in enumerate(list(audio_results.keys()) or [f"v{j}" for j in range(3)]):
                    audio_results[vid_id] = map_to_affective_axes(synth[i])

            # Compute cross-modal discord score for each video
            for vid_id, audio_aff in audio_results.items():
                if audio_aff is None:
                    continue
                vis_aff = video_level_scores.get(vid_id)
                if vis_aff is None:
                    # Try matching by partial name
                    for k in video_level_scores:
                        if vid_id in k or k in vid_id:
                            vis_aff = video_level_scores[k]
                            break
                if vis_aff:
                    discord = cross_modal_discord_score(audio_aff, vis_aff)
                    print(f"  {vid_id}: audio-visual discord = {discord:.4f} "
                          f"({'⚠ high mismatch' if discord > 0.8 else '✓ aligned'})")
                    manifest.record(f"audio_discord_{vid_id}", discord=discord)
                else:
                    print(f"  {vid_id}: audio scores computed but no visual scores available for discord.")
        else:
            # Demo mode: generate synthetic audio and show discord for a synthetic visual
            print("  No .mp4 paths in index — running audio demo with synthetic data…")
            synth_feats = generate_synthetic_audio_features(1, seed=0)[0]
            audio_aff = map_to_affective_axes(synth_feats)
            synth_visual = {ax: 0.0 for ax in audio_aff}  # neutral visual
            discord = cross_modal_discord_score(audio_aff, synth_visual)
            print(f"  Demo audio affective: {audio_aff}")
            print(f"  Demo audio-visual discord (vs neutral visual): {discord:.4f}")
            manifest.record("audio_discord_demo", discord=discord)

    except (ImportError, OSError) as exc:
        logger.warning("Audio features step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 19 – Local embedding vector store (ANN index demo)
    # -----------------------------------------------------------------------
    print("\n── Step 19: Embedding vector store (ANN search demo) ──────────────")
    try:
        from src.vector_store import EmbeddingVectorStore

        if embeddings is not None and len(embeddings) > 0:
            store = EmbeddingVectorStore(metric="cosine")
            store.add(embeddings, index)
            print(f"  Added {len(store)} frame embeddings to vector store (D={embeddings.shape[1]}).")

            # Demo: query with 3 frames and print top-5 ANN hits
            query_count = min(3, len(store))
            for qi in range(query_count):
                results = store.search(embeddings[qi], k=5)
                top_ids = [r.idx for r in results]
                top_scores = [f"{r.score:.4f}" for r in results]
                print(f"  Query frame {qi}: top-5 ANN hits = {top_ids}  "
                      f"scores = {top_scores}")

            # Save the store
            vs_path = os.path.join(OUTPUTS_DIR, "vector_store")
            store.save(vs_path)
            print(f"  Vector store saved: {vs_path}.npz + {vs_path}.meta.json")
            manifest.record(
                "vector_store",
                n_vectors=len(store),
                dim=int(embeddings.shape[1]),
            )
            manifest.add_artifact(f"{vs_path}.npz", "EmbeddingVectorStore compressed")
        else:
            print("  No embeddings available; skipping vector store step.")

    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Vector store step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 14 – Save run manifest
    # -----------------------------------------------------------------------
    manifest_path = os.path.join(OUTPUTS_DIR, "run_manifest.json")
    manifest.save(manifest_path)
    print(f"\n  Run manifest: {os.path.abspath(manifest_path)}")
    print(f"\n{manifest.summary()}")
    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
