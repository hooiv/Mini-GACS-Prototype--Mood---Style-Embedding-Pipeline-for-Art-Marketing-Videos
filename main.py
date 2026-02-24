"""
main.py
-------
End-to-end pipeline orchestrator for the Mini GACS
Mood & Style Embedding Pipeline.

Steps
-----
1.  (Optional) Download sample videos.
2.  Extract frames from all videos found in ``data/videos/``.
3.  Compute CLIP embeddings for all frames.
3b. Content-adaptive deduplication — remove near-duplicate frames.
4.  Calculate pairwise cosine-similarity matrix.
5.  Run top-5 retrieval for at least 3 query frames.
6.  Generate visualisations (heatmap, bar chart, retrieval grids).
7.  Write a Markdown similarity report.
8.  Affective scoring (single-prompt + multi-prompt ensemble).
9.  Vibe clustering with silhouette-based quality metrics.
10. Temporal analysis with consecutive-frame transition detection.
11. Vibe–performance regression (synthetic CTR prediction).
12. Write structured run manifest (run_manifest.json).

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
from typing import Optional

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
from src.affective_scoring import AffectiveScorer, DEFAULT_AXES, MULTI_PROMPT_AXES
from src.clustering import VibeClusterer, auto_n_clusters
from src.temporal_analysis import TemporalAnalyser
from src.performance_predictor import (
    generate_synthetic_performance_data,
    build_feature_names,
    VibePerformancePredictor,
)
from src.frame_deduplication import deduplicate_frames
from src.experiment_manifest import PipelineManifest

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
    p.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        metavar="K",
        help="Number of vibe clusters (default: 0 = auto-detect via silhouette).",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Initialise run manifest — collects all metrics for reproducibility
    manifest = PipelineManifest(
        config={
            "model": args.model,
            "interval_seconds": args.interval,
            "max_frames": args.max_frames,
            "dedup_threshold": args.dedup_threshold,
            "n_clusters_arg": args.n_clusters,
        }
    )

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

    manifest.record("similarity", **stats, n_frames=len(index))

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

        # Update frame_scores to use ensemble for downstream modules
        frame_scores = ens_scores

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
            n_pca = features.shape[1] - len(frame_scores)
            feat_names = build_feature_names(frame_scores, n_pca)

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

    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Performance regression step failed (%s); continuing.", exc)

    # -----------------------------------------------------------------------
    # Step 12 – Save run manifest
    # -----------------------------------------------------------------------
    manifest_path = os.path.join(OUTPUTS_DIR, "run_manifest.json")
    manifest.save(manifest_path)
    print(f"\n  Run manifest: {os.path.abspath(manifest_path)}")
    print(f"\n{manifest.summary()}")
    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
