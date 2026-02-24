# Mini GACS Prototype — Mood & Style Embedding Pipeline for Art/Marketing Videos

A self-contained Python pipeline that:

1. **Ingests** 2–3 short public-domain art/marketing videos.
2. **Extracts** representative frames at a configurable time interval.
3. **Embeds** each frame with OpenAI CLIP (via HuggingFace *transformers*).
4. **Computes** pairwise cosine-similarity scores to capture the visual "vibe"
   across frames.
5. **Scores** frames on named affective axes (energy, warmth, complexity, …)
   using zero-shot text-guided CLIP probing.
6. **Clusters** frames by visual vibe using K-means + PCA projection.
7. **Visualises** results as heatmaps, retrieval grids, radar charts, scatter
   plots, and a Markdown report.

---

## Repository Structure

```
.
├── src/
│   ├── frame_extractor.py    # Video loading, frame extraction, metadata I/O
│   ├── embeddings.py         # CLIP embedding computation and persistence (+ encode_text)
│   ├── similarity.py         # Cosine-similarity matrix, top-k retrieval, memory-efficient retrieval
│   ├── visualization.py      # Matplotlib heatmap, grids, bar chart, report
│   ├── affective_scoring.py  # Zero-shot text-guided affective axis scoring (multi-prompt ensemble)
│   ├── clustering.py         # K-means vibe clustering + PCA/t-SNE scatter (silhouette quality)
│   ├── temporal_analysis.py  # Temporal similarity curve, consecutive-sim scene transitions, pacing
│   ├── performance_predictor.py  # Vibe → CTR/ROAS ridge/MLP regression (CV leakage-free)
│   ├── frame_deduplication.py    # Greedy cosine-threshold dedup + MMR diverse selection
│   ├── quality_filter.py         # Technical frame quality scoring: blur, exposure, luminance
│   ├── experiment_manifest.py    # Structured run manifest for reproducibility
│   ├── ranking.py                # Diversity-constrained MMR creative ranking with bootstrap CI
│   ├── calibration.py            # Post-hoc Platt/isotonic predictor calibration + ECE
│   ├── text_query.py             # Text-guided creative retrieval (CLIP text-to-image search)
│   ├── drift_detector.py         # Embedding distribution drift monitoring (MMD + KS + IF)
│   ├── explainability.py         # Occlusion saliency: spatial importance maps for affective scores
│   └── ab_testing.py             # Bayesian A/B testing: P(A>B), Thompson sampling, tournament
├── tests/
│   └── test_pipeline.py     # Unit + integration tests (pytest, 216 tests)
├── data/
│   ├── videos/              # Source videos (downloaded or user-provided)
│   ├── frames/              # Extracted frame images (auto-generated)
│   ├── metadata/            # CSV / JSON metadata files (auto-generated)
│   └── embeddings/          # Saved .npy embedding arrays (auto-generated)
├── outputs/                 # All generated visualisations and reports
├── conftest.py              # Shared pytest fixtures (synthetic video, mock CLIP)
├── pytest.ini               # Pytest discovery configuration
├── notebook.ipynb           # Interactive Jupyter walkthrough (all steps)
├── download_videos.py       # Helper: download 3 sample CC0 videos
├── main.py                  # End-to-end pipeline orchestrator (16 steps)
├── requirements.txt
├── README.md                # This file
└── REPORT.md                # GenTA / GACS design discussion
```

---

## Quick Start

### 1 · Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
```

> **CPU-only PyTorch** (faster to install, no GPU required):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 2 · Download sample videos (optional – or bring your own)

```bash
python download_videos.py --output-dir data/videos
```

The downloader fetches three short CC0 clips from Wikimedia Commons with
automatic fallbacks.  Alternatively, drop any `.mp4 / .webm / .avi` files
into `data/videos/` yourself.

### 3 · Run the full pipeline

```bash
python main.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--interval SECONDS` | `1.0` | Seconds between extracted frames |
| `--max-frames N` | `50` | Upper bound on frames per video |
| `--model NAME` | `openai/clip-vit-base-patch32` | HuggingFace CLIP variant |
| `--top-k N` | `5` | Similar frames per query |
| `--n-queries N` | `3` | Number of query frames |
| `--n-clusters K` | `0` | Vibe clusters (0 = auto-detect) |
| `--skip-download` | – | Skip video download step |
| `--skip-extraction` | – | Reuse existing frames/metadata |
| `--skip-embedding` | – | Reuse existing `.npy` embeddings |

### 4 · Inspect outputs

After a successful run, `outputs/` contains:

| File | Description |
|------|-------------|
| `similarity_heatmap.png` | Full pairwise cosine-similarity heatmap |
| `cross_video_similarity_bar.png` | Within-video vs cross-video mean similarity |
| `top_k_query_<N>.png` | Retrieval grid for each query frame |
| `similarity_report.md` | Markdown table of top-5 retrievals per query |
| `affective_heatmap.png` | Affective axis scores heatmap (axes × frames) |
| `affective_radar.png` | Per-video affective profile radar chart |
| `affective_scores.json` | Frame-level affective scores (JSON) |
| `vibe_cluster_scatter.png` | 2-D PCA scatter of frames coloured by vibe cluster |
| `cluster_assignments.json` | Per-frame cluster label assignments (JSON) |
| `narrative_arc.png` | Temporal similarity curve with scene transitions marked |
| `pacing_comparison.png` | Pacing & coherence bar chart comparing videos |
| `temporal_stats.json` | Per-frame temporal similarity scores (JSON) |
| `predictor_cv_results.png` | Spearman ρ per CV fold bar chart |
| `predictor_feature_importance.png` | Top feature importances (vibe → CTR) |
| `predictor_predicted_vs_actual.png` | Scatter of predicted vs actual CTR |
| `predictor_model.json` | Ridge model coefficients and scaler parameters |

### 5 · Interactive Jupyter notebook

```bash
jupyter notebook notebook.ipynb
```

Runs all eleven pipeline steps interactively, showing inline visualisations
at each stage.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite (184 tests) uses synthetic videos and a mocked CLIP model so
**no GPU and no internet connection are needed**.  All tests pass in under
60 seconds.

---

## Module Overview

### `src/frame_extractor.py`

| Function | Purpose |
|----------|---------|
| `extract_frames(video_path, output_dir, interval_seconds, max_frames)` | Extract frames from a single video |
| `save_metadata(metadata, output_path, fmt)` | Persist metadata as CSV or JSON |
| `load_metadata(metadata_path)` | Reload saved metadata |
| `process_video_directory(...)` | Batch-process all videos in a directory |

### `src/embeddings.py`

| Symbol | Purpose |
|--------|---------|
| `EmbeddingModel` | CLIP wrapper with batched `embed_images()` |
| `save_embeddings / load_embeddings` | NumPy `.npy` + JSON index persistence |
| `compute_and_save_embeddings` | High-level convenience function |

### `src/similarity.py`

| Function | Purpose |
|----------|---------|
| `cosine_similarity_matrix(embeddings)` | Full ``(N×N)`` similarity matrix |
| `get_top_k_similar(query_idx, ...)` | Top-k nearest neighbours for one query |
| `batch_top_k_queries(query_indices, ...)` | Run multiple queries at once |
| `compute_inter_video_stats(...)` | Within/cross-video similarity statistics |

### `src/visualization.py`

| Function | Purpose |
|----------|---------|
| `plot_similarity_heatmap(...)` | Seaborn-style heatmap PNG |
| `plot_top_k_grid(...)` | Query + neighbours image grid |
| `plot_cross_video_similarity_bar(...)` | Bar chart comparing similarity groups |
| `generate_similarity_report(...)` | Markdown report with tables |

### `src/affective_scoring.py`

| Symbol | Purpose |
|--------|---------|
| `AffectiveScorer` | Zero-shot text-guided CLIP affective scorer |
| `AffectiveScorer.score_frames(embeddings)` | Frame-level axis scores ``(N,)`` per axis |
| `AffectiveScorer.score_video_level(scores, index)` | Mean-pooled per-video profile |
| `AffectiveScorer.plot_heatmap(...)` | Axes × frames score heatmap |
| `AffectiveScorer.plot_radar(...)` | Per-video radar chart |
| `AffectiveScorer.save_scores(...)` | JSON export of frame-level scores |
| `DEFAULT_AXES` | Built-in 6-axis affective taxonomy |

### `src/clustering.py`

| Symbol | Purpose |
|--------|---------|
| `VibeClusterer` | K-means + PCA/t-SNE clustering of frame embeddings |
| `VibeClusterer.fit(embeddings)` | Fit K-means, return label array ``(N,)`` |
| `VibeClusterer.predict(embeddings)` | Assign new frames to nearest cluster |
| `VibeClusterer.project_2d(embeddings)` | PCA or t-SNE 2-D projection |
| `VibeClusterer.plot_scatter(...)` | 2-D scatter plot coloured by cluster |
| `VibeClusterer.cluster_summary(labels, index)` | Per-cluster size and video distribution |
| `VibeClusterer.save_cluster_assignments(...)` | JSON export of per-frame labels |
| `auto_n_clusters(embeddings)` | Elbow-heuristic k suggestion |

### `src/temporal_analysis.py`

| Symbol | Purpose |
|--------|---------|
| `TemporalAnalyser` | Temporal visual-style dynamics analyser |
| `TemporalAnalyser.compute_temporal_curve(embeddings, index)` | Per-frame neighbourhood similarity ``(N,)`` |
| `TemporalAnalyser.detect_scene_transitions(curve, threshold)` | Indices of sharp style-change frames |
| `TemporalAnalyser.pacing_score(curve)` | Variance of temporal curve (high = dynamic editing) |
| `TemporalAnalyser.coherence_score(curve)` | Mean of temporal curve (high = smooth narrative) |
| `TemporalAnalyser.per_video_stats(curve, index)` | Coherence, pacing, transitions per video |
| `TemporalAnalyser.plot_narrative_arc(...)` | Line chart of temporal curve + transition markers |
| `TemporalAnalyser.plot_pacing_comparison(...)` | Bar chart of pacing & coherence across videos |
| `TemporalAnalyser.save_temporal_stats(...)` | JSON export of per-frame temporal scores |

### `src/performance_predictor.py`

| Symbol | Purpose |
|--------|---------|
| `generate_synthetic_performance_data(...)` | Synthetic CTR/ROAS labels from affective + PCA features |
| `build_feature_names(affective_scores, n_pca)` | Consistent feature name list |
| `VibePerformancePredictor` | Ridge / MLP predictor: vibe features → CTR/ROAS |
| `VibePerformancePredictor.fit(features, labels)` | Train the predictor |
| `VibePerformancePredictor.predict(features)` | Score new creatives |
| `VibePerformancePredictor.cross_validate(...)` | K-fold Spearman ρ + RMSE evaluation |
| `VibePerformancePredictor.feature_importance(...)` | Top-k drivers of predicted performance |
| `VibePerformancePredictor.plot_cv_results(...)` | CV fold bar chart |
| `VibePerformancePredictor.plot_feature_importance(...)` | Importance horizontal bar chart |
| `VibePerformancePredictor.plot_predicted_vs_actual(...)` | Scatter plot of predicted vs actual |
| `VibePerformancePredictor.save_model(...)` | JSON export of model coefficients |

---

## Design Notes

- All embeddings are **L2-normalised** before storage, so cosine similarity
  reduces to a simple dot product (fast, numerically stable).
- Frames are skipped gracefully when files are missing or unreadable; the
  pipeline never crashes due to a single bad file.
- Intermediate artefacts (frames, metadata, embeddings) are written to disk
  at every stage so the pipeline is fully restartable with `--skip-*` flags.
- The CLIP model is loaded once and reused for all batches to minimise
  memory overhead.
- The performance predictor uses **Spearman ρ** as the evaluation metric — the
  correct choice for ad-creative ranking systems where relative order matters
  more than absolute values.

---

## See Also

- [`REPORT.md`](REPORT.md) — GenTA perspective, AI tool usage notes, and
  next steps for connecting the engine to performance metrics (CTR/ROAS).
