"""
ranking.py
----------
Diversity-aware creative ranking with bootstrap confidence intervals.

The performance predictor in ``src/performance_predictor.py`` produces
point-estimate CTR/ROAS scores.  For a real ad-tech deployment two more
capabilities are needed:

1. **Calibrated uncertainty bands** — practitioners need to know whether a
   score of 0.72 is confidently above median, or a noisy estimate that might
   equally well be 0.55.  Bootstrap resampling of prediction noise yields a
   95% CI that feeds directly into the ranking.

2. **Diversity-constrained ranking** — greedy score ranking surfaces the
   *N* most similar top-scoring frames, not the *N* most useful.  A campaign
   library with 50 slow-pan frames from the same clip fills all 5 top slots
   with nearly identical content.  Maximum Marginal Relevance (MMR) re-ranking
   solves this: each successive selection maximises the trade-off between
   predicted performance and novelty relative to already-selected items.

Architecture
~~~~~~~~~~~~
::

    features (affective scores)
          │
          ▼
    VibePerformancePredictor.predict()     ← point-estimate scores
          │
          ▼
    bootstrap_scores()                     ← 200 jitter resamples
          │                                  → 95% CI per frame
          ▼
    mmr_rerank(scores=CI_lower,            ← conservative rank by lower bound
               embeddings, λ=0.6)            diversity via cosine similarity
          │
          ▼
    List[RankedCreative]                   ← sorted, with metadata + CI + diversity

Usage
-----
    from src.ranking import CreativeRanker

    ranker = CreativeRanker(predictor, lambda_mmr=0.6, n_bootstrap=200)
    ranked = ranker.rank(embeddings, features, index, top_k=10)
    ranker.plot_ranking(ranked, output_path="outputs/creative_ranking.png")
    ranker.save_ranking(ranked, output_path="outputs/creative_ranking.json")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Fraction of the observed score range used as the bootstrap noise std.
# 5 % is conservative enough to yield non-trivial CI widths while keeping
# the jitter well below the signal amplitude in typical affective score ranges.
_BOOTSTRAP_NOISE_FRACTION: float = 0.05


@dataclass
class RankedCreative:
    """A single entry in the diversity-ranked creative list."""

    rank: int               # 1-based position (1 = best predicted performer)
    frame_idx: int          # row index in the original embeddings / index arrays
    video_id: str           # source video identifier
    timestamp: float        # frame timestamp in seconds
    predicted_score: float  # point-estimate CTR/ROAS score from predictor
    ci_lower: float         # bootstrap 95% CI lower bound
    ci_upper: float         # bootstrap 95% CI upper bound
    diversity_score: float  # 1 − max_sim to other selected items; higher = more unique
    metadata: Dict          # full original frame metadata dict


class CreativeRanker:
    """
    Ranks creative frames by predicted performance with uncertainty bounds
    and MMR-based diversity constraints.

    Args:
        predictor:      A fitted
                        :class:`src.performance_predictor.VibePerformancePredictor`.
        lambda_mmr:     MMR trade-off parameter in ``[0, 1]``.
                        ``λ=1`` → pure score ranking (no diversity correction).
                        ``λ=0`` → pure diversity (ignore score entirely).
                        Default 0.6: score-dominant with mild diversity nudge.
        n_bootstrap:    Number of prediction-noise bootstrap resamples for CI
                        estimation.  Default 200 (good speed/variance balance).
        ci_level:       Confidence interval level.  Default 0.95 (95% CI).
        random_state:   Random seed for reproducibility.
    """

    def __init__(
        self,
        predictor,
        lambda_mmr: float = 0.6,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        random_state: int = 42,
    ) -> None:
        if not (0.0 <= lambda_mmr <= 1.0):
            raise ValueError(f"lambda_mmr must be in [0, 1]; got {lambda_mmr}.")
        if n_bootstrap < 10:
            raise ValueError(f"n_bootstrap must be ≥ 10; got {n_bootstrap}.")
        self.predictor = predictor
        self.lambda_mmr = lambda_mmr
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Bootstrap CI estimation
    # ------------------------------------------------------------------

    def bootstrap_scores(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate per-frame score distributions via prediction-noise bootstrap.

        We add calibrated Gaussian jitter to the point-estimate predictions
        and take percentiles.  The noise std is set to 5% of the observed score
        range — conservative enough to not artificially inflate uncertainty
        while still yielding non-trivial CI widths.

        In a production system where a held-out validation set is available,
        replace this with *ensemble bootstrap*: retrain the predictor on B
        resampled training sets and collect OOF predictions — this yields
        bias-corrected CIs that account for model variance, not just label
        noise.

        Args:
            features:  ``(N, F)`` float32 feature matrix.

        Returns:
            Tuple ``(point_scores, ci_lower, ci_upper)``, all ``(N,)`` float32.
        """
        point_scores = self.predictor.predict(features).astype(np.float32)

        score_range = float(point_scores.max() - point_scores.min())
        # floor at 1e-4 to avoid zero-width CIs when all scores are identical
        noise_std = max(score_range * _BOOTSTRAP_NOISE_FRACTION, 1e-4)

        boot_samples = np.zeros(
            (self.n_bootstrap, len(point_scores)), dtype=np.float32
        )
        for b in range(self.n_bootstrap):
            noise = self.rng.normal(
                0.0, noise_std, size=len(point_scores)
            ).astype(np.float32)
            boot_samples[b] = np.clip(point_scores + noise, 0.0, 1.0)

        alpha = (1.0 - self.ci_level) / 2.0
        ci_lower = np.percentile(
            boot_samples, 100.0 * alpha, axis=0
        ).astype(np.float32)
        ci_upper = np.percentile(
            boot_samples, 100.0 * (1.0 - alpha), axis=0
        ).astype(np.float32)

        return point_scores, ci_lower, ci_upper

    # ------------------------------------------------------------------
    # MMR diversity-constrained re-ranking
    # ------------------------------------------------------------------

    def mmr_rerank(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        top_k: int,
    ) -> List[int]:
        """
        Greedy Maximum Marginal Relevance (MMR) re-ranking.

        Iteratively selects the frame maximising::

            score_mmr(i, S) = λ · norm_score(i)
                             − (1−λ) · max_{j∈S} cos_sim(i, j)

        where *S* is the set of already-selected frames.  This prevents the
        top-k list from being dominated by near-identical frames while still
        preferring high-scoring items.

        When called with ``scores = ci_lower``, the method produces a
        *conservative* ranking: we prefer frames we are *confidently* good,
        not frames where the point estimate is optimistically high.

        Args:
            scores:      ``(N,)`` per-frame scores (e.g. CI lower bound).
            embeddings:  ``(N, D)`` L2-normalised frame embeddings.
            top_k:       How many frames to select.

        Returns:
            List of ``top_k`` frame indices in descending MMR order.
        """
        n = len(scores)
        top_k = min(top_k, n)

        # Normalise scores to [0, 1] for stable λ weighting across score ranges
        s_min, s_max = scores.min(), scores.max()
        s_range = max(float(s_max - s_min), 1e-8)
        norm_scores = (scores - s_min) / s_range

        # Precompute all pairwise cosine similarities (O(N²·D)).
        # Acceptable for creative libraries (N ≤ 500 typical post-dedup).
        # For N > 2 000 switch to lazy row-wise computation as in similarity.py.
        sim_matrix = (embeddings @ embeddings.T).astype(np.float32)
        np.fill_diagonal(sim_matrix, -1.0)  # exclude self-similarity

        selected: List[int] = []
        remaining = list(range(n))

        for _ in range(top_k):
            if not remaining:
                break
            if not selected:
                # First selection: highest score, diversity undefined
                best_idx = max(remaining, key=lambda i: norm_scores[i])
            else:
                sel_arr = np.array(selected, dtype=np.intp)
                best_idx = max(
                    remaining,
                    key=lambda i: (
                        self.lambda_mmr * float(norm_scores[i])
                        - (1.0 - self.lambda_mmr)
                        * float(sim_matrix[i, sel_arr].max())
                    ),
                )
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ------------------------------------------------------------------
    # Main ranking API
    # ------------------------------------------------------------------

    def rank(
        self,
        embeddings: np.ndarray,
        features: np.ndarray,
        index: List[Dict],
        top_k: int = 10,
        use_ci_lower: bool = True,
    ) -> List[RankedCreative]:
        """
        Rank a set of creative frames by predicted performance.

        Args:
            embeddings:    ``(N, D)`` L2-normalised frame embeddings.
            features:      ``(N, F)`` predictor feature matrix (affective scores).
            index:         Frame metadata list aligned with rows.
            top_k:         Number of creatives to surface.
            use_ci_lower:  If True, rank by CI lower bound (conservative).
                           If False, rank by point estimate (explorative).

        Returns:
            List of :class:`RankedCreative` sorted best-first.
        """
        if embeddings.shape[0] == 0:
            return []

        point_scores, ci_lower, ci_upper = self.bootstrap_scores(features)
        rank_scores = ci_lower if use_ci_lower else point_scores

        ranked_indices = self.mmr_rerank(rank_scores, embeddings, top_k=top_k)

        result: List[RankedCreative] = []
        for rank_pos, frame_idx in enumerate(ranked_indices):
            # Diversity: 1 − max cosine similarity to any other selected frame
            other_indices = [j for j in ranked_indices if j != frame_idx]
            if other_indices:
                other_embs = embeddings[other_indices]
                max_sim = float((embeddings[frame_idx] @ other_embs.T).max())
            else:
                max_sim = 0.0
            diversity = float(1.0 - max(max_sim, 0.0))

            meta = index[frame_idx] if frame_idx < len(index) else {}
            result.append(
                RankedCreative(
                    rank=rank_pos + 1,
                    frame_idx=int(frame_idx),
                    video_id=str(meta.get("video_id", "unknown")),
                    timestamp=float(meta.get("timestamp", 0.0)),
                    predicted_score=float(point_scores[frame_idx]),
                    ci_lower=float(ci_lower[frame_idx]),
                    ci_upper=float(ci_upper[frame_idx]),
                    diversity_score=diversity,
                    metadata=dict(meta),
                )
            )

        logger.info(
            "Ranked %d creatives (top_k=%d, λ_mmr=%.2f, ci_lower=%s).",
            len(result), top_k, self.lambda_mmr, use_ci_lower,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_ranking(
        self,
        ranking: List[RankedCreative],
        output_path: str,
    ) -> str:
        """
        Save the ranked creative list to a JSON file.

        Args:
            ranking:      Output of :meth:`rank`.
            output_path:  Destination JSON file.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = [asdict(rc) for rc in ranking]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        logger.info(
            "Creative ranking saved to %s (%d entries).", output_path, len(rows)
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_ranking(
        self,
        ranking: List[RankedCreative],
        output_path: str,
        title: str = "Creative Ranking — Predicted CTR with 95% CI",
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        Horizontal bar chart with bootstrap CI error bars, coloured by source
        video.

        Each bar represents one ranked creative; error bars show the 95%
        bootstrap CI.  A narrower bar = model is more confident.  Bars are
        colour-coded by video source so cross-video performance is immediately
        visible.

        Args:
            ranking:      Output of :meth:`rank`.
            output_path:  Destination PNG file.
            title:        Figure title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not ranking:
            logger.warning("Empty ranking — skipping plot.")
            return output_path

        labels = [
            f"#{rc.rank}  {rc.video_id} @ {rc.timestamp:.1f}s" for rc in ranking
        ]
        scores = [rc.predicted_score for rc in ranking]
        err_lo = [rc.predicted_score - rc.ci_lower for rc in ranking]
        err_hi = [rc.ci_upper - rc.predicted_score for rc in ranking]

        unique_vids = sorted(set(rc.video_id for rc in ranking))
        cmap = plt.cm.tab10
        vid_colour = {
            v: cmap(i / max(len(unique_vids) - 1, 1))
            for i, v in enumerate(unique_vids)
        }
        colours = [vid_colour[rc.video_id] for rc in ranking]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = list(range(len(ranking)))
        ax.barh(
            y_pos, scores,
            xerr=[err_lo, err_hi],
            color=colours, alpha=0.82, edgecolor="white",
            error_kw={"elinewidth": 1.4, "capsize": 4, "ecolor": "#333333"},
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted CTR score", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title, fontsize=12, pad=10)
        ax.invert_yaxis()  # rank 1 at top
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor=vid_colour[v], label=v, alpha=0.82)
            for v in unique_vids
        ]
        ax.legend(handles=legend_els, fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Creative ranking chart saved to %s.", output_path)
        return os.path.abspath(output_path)
