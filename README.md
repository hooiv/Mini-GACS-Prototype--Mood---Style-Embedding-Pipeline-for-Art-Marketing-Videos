# Mini GACS Prototype — Mood & Style Embedding Pipeline for Art/Marketing Videos

A self-contained Python pipeline that:

1. **Ingests** 2–3 short public-domain art/marketing videos.
2. **Extracts** representative frames at a configurable time interval.
3. **Embeds** each frame with OpenAI CLIP (via HuggingFace *transformers*).
4. **Computes** pairwise cosine-similarity scores to capture the visual "vibe"
   across frames.
5. **Visualises** results as a heatmap, retrieval grids, and a Markdown report.

---

## Repository Structure

```
.
├── src/
│   ├── frame_extractor.py   # Video loading, frame extraction, metadata I/O
│   ├── embeddings.py        # CLIP embedding computation and persistence
│   ├── similarity.py        # Cosine-similarity matrix and top-k retrieval
│   └── visualization.py     # Matplotlib heatmap, grids, bar chart, report
├── tests/
│   └── test_pipeline.py     # Unit + integration tests (pytest)
├── data/
│   ├── videos/              # Source videos (downloaded or user-provided)
│   ├── frames/              # Extracted frame images (auto-generated)
│   ├── metadata/            # CSV / JSON metadata files (auto-generated)
│   └── embeddings/          # Saved .npy embedding arrays (auto-generated)
├── outputs/                 # Heatmap PNG, bar chart, retrieval grids, report
├── download_videos.py       # Helper: download 3 sample CC0 videos
├── main.py                  # End-to-end pipeline orchestrator
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
| `--skip-download` | – | Skip video download step |
| `--skip-extraction` | – | Reuse existing frames/metadata |
| `--skip-embedding` | – | Reuse existing `.npy` embeddings |

### 4 · Inspect outputs

After a successful run, `outputs/` contains:

- `similarity_heatmap.png` — full pairwise cosine-similarity heatmap
- `cross_video_similarity_bar.png` — within-video vs cross-video mean similarity
- `top_k_query_<N>.png` — retrieval grid for each query frame
- `similarity_report.md` — Markdown table of top-5 retrievals per query

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite uses synthetic videos and a mocked CLIP model so **no GPU
and no internet connection are needed**.  All tests should pass in under
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

---

## See Also

- [`REPORT.md`](REPORT.md) — GenTA perspective, AI tool usage notes, and
  next steps for connecting the engine to performance metrics (CTR/ROAS).
