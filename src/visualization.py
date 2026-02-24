"""
visualization.py
----------------
Generates similarity heatmaps and retrieval-result reports using Matplotlib.
All functions write output to disk *and* return the figure/path so callers
can inspect or further customise them.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend so plots can be generated in headless
# environments (CI, server, etc.)
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    index: List[Dict],
    output_path: str,
    title: str = "Frame Similarity Heatmap (Cosine)",
    figsize: Tuple[int, int] = (14, 12),
    cmap: str = "viridis",
    max_labels: int = 40,
) -> str:
    """
    Plot and save a heatmap of the pairwise cosine-similarity matrix.

    Tick labels are abbreviated to ``<video_id>/<frame_idx>`` to keep the
    plot readable.  When there are more than *max_labels* frames the ticks
    are hidden to avoid clutter.

    Args:
        similarity_matrix: ``(N, N)`` float32 array.
        index:             Metadata list aligned with rows.
        output_path:       Destination file for the PNG image.
        title:             Figure title.
        figsize:           ``(width, height)`` in inches.
        cmap:              Matplotlib colourmap.
        max_labels:        Maximum number of axis tick labels shown.

    Returns:
        Absolute path to the saved PNG file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n = similarity_matrix.shape[0]
    labels = [
        f"{entry.get('video_id', '?')}/{entry.get('frame_idx', i)}"
        for i, entry in enumerate(index)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    if n <= max_labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} frames (tick labels hidden for clarity)")
        ax.set_ylabel(f"{n} frames (tick labels hidden for clarity)")

    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Heatmap saved to %s.", output_path)
    return os.path.abspath(output_path)


def plot_top_k_grid(
    query_idx: int,
    top_k_results: List[Dict],
    index: List[Dict],
    output_path: str,
    max_cols: int = 6,
) -> str:
    """
    Display the query frame alongside its top-k nearest neighbours in a grid.

    Args:
        query_idx:      Row index of the query frame.
        top_k_results:  Output of :func:`similarity.get_top_k_similar`.
        index:          Full metadata list.
        output_path:    Destination PNG file.
        max_cols:       Maximum images per row.

    Returns:
        Absolute path to the saved PNG.
    """
    from PIL import Image  # local import – optional dependency

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Collect items to display: query first, then results
    items = [{"meta": index[query_idx], "label": "QUERY", "border": "red"}]
    for r in top_k_results:
        label = f"sim={r['similarity']:.3f}\n{r.get('video_id','?')}/f{r.get('frame_idx','?')}"
        items.append({"meta": r, "label": label, "border": "blue"})

    n_items = len(items)
    n_cols = min(n_items, max_cols)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for flat_idx, item in enumerate(items):
        r, c = divmod(flat_idx, n_cols)
        ax = axes[r, c]
        fpath = item["meta"].get("file_path", "")
        if fpath and os.path.exists(fpath):
            try:
                img = Image.open(fpath).convert("RGB")
                ax.imshow(np.array(img))
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "load error", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(item["label"], fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(item["border"])
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for flat_idx in range(n_items, n_rows * n_cols):
        r, c = divmod(flat_idx, n_cols)
        axes[r, c].set_visible(False)

    plt.suptitle(
        f"Query frame {query_idx} and top-{len(top_k_results)} similar frames",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Top-k grid saved to %s.", output_path)
    return os.path.abspath(output_path)


def generate_similarity_report(
    similarity_matrix: np.ndarray,
    index: List[Dict],
    query_results: Dict[int, List[Dict]],
    stats: Dict[str, float],
    output_path: str,
) -> str:
    """
    Write a human-readable text/Markdown report summarising similarity
    results.

    Args:
        similarity_matrix: ``(N, N)`` similarity matrix.
        index:             Metadata list.
        query_results:     Dict from :func:`similarity.batch_top_k_queries`.
        stats:             Dict from :func:`similarity.compute_inter_video_stats`.
        output_path:       Destination ``.md`` or ``.txt`` file.

    Returns:
        Absolute path to the written report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    lines = [
        "# Vibe Similarity Report",
        "",
        "## Dataset Overview",
        f"- Total frames embedded: **{len(index)}**",
        f"- Embedding matrix shape: ``{similarity_matrix.shape}``",
        "",
        "## Aggregate Similarity Statistics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall mean cosine similarity | {stats.get('overall_mean', float('nan')):.4f} |",
        f"| Overall median cosine similarity | {stats.get('overall_median', float('nan')):.4f} |",
        f"| Within-video mean similarity | {stats.get('within_video_mean', float('nan')):.4f} |",
        f"| Cross-video mean similarity | {stats.get('cross_video_mean', float('nan')):.4f} |",
        "",
        "## Top-5 Retrieval Results per Query Frame",
        "",
    ]

    for qidx, results in query_results.items():
        meta = index[qidx] if qidx < len(index) else {}
        lines.append(
            f"### Query: frame {qidx} "
            f"(`{meta.get('video_id','?')}` @ {meta.get('timestamp','?')}s)"
        )
        lines.append("")
        lines.append("| Rank | Video ID | Frame Idx | Timestamp (s) | Similarity |")
        lines.append("|------|----------|-----------|---------------|------------|")
        for rank, r in enumerate(results, 1):
            lines.append(
                f"| {rank} | {r.get('video_id','?')} | {r.get('frame_idx','?')} "
                f"| {r.get('timestamp','?')} | {r.get('similarity', 0):.4f} |"
            )
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    logger.info("Similarity report written to %s.", output_path)
    return os.path.abspath(output_path)


def plot_cross_video_similarity_bar(
    stats: Dict[str, float],
    output_path: str,
) -> str:
    """
    Simple bar chart comparing within-video vs cross-video mean similarity.

    Args:
        stats:        Dict from :func:`similarity.compute_inter_video_stats`.
        output_path:  Destination PNG file.

    Returns:
        Absolute path to the saved PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    categories = ["Within-video", "Cross-video", "Overall"]
    values = [
        stats.get("within_video_mean", 0.0),
        stats.get("cross_video_mean", 0.0),
        stats.get("overall_mean", 0.0),
    ]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, values, color=colours, edgecolor="black", width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Pairwise Cosine Similarity by Frame Pair Type")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Bar chart saved to %s.", output_path)
    return os.path.abspath(output_path)
