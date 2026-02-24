"""
ab_testing.py
-------------
Bayesian A/B testing and Thompson sampling for creative performance comparison.

Problem
~~~~~~~
``VibePerformancePredictor`` and ``CreativeRanker`` produce point-estimate CTR/ROAS
scores plus 95% bootstrap confidence intervals.  But practitioners need answers to
concrete decision questions:

    "Is creative A significantly better than creative B?"
    "Which 3 creatives should we serve to maximise CTR this week?"
    "Are we confident enough to eliminate the bottom performers?"

These require **statistical comparison under uncertainty**, not just scalar ranking.
This module provides two complementary tools:

1. **Closed-form Bayesian pairwise test** — computes P(A > B) analytically under
   a Gaussian score model.  Fast, exact, and requires only point estimates + CIs.

2. **Thompson sampling** — Monte Carlo multi-armed bandit selection.  Draws
   N_samples from each creative's score posterior and counts wins.  Handles > 2
   creatives naturally and provides a frequency-domain check on the closed-form
   result.

Gaussian model
~~~~~~~~~~~~~~
Under the normal approximation to the bootstrap CI::

    score_A ~ N(μ_A, σ_A)
    σ_A     = (ci_upper_A − ci_lower_A) / (2 · 1.96)   [95 % CI → σ]

    score_B ~ N(μ_B, σ_B)

    P(A > B) = P(X_A − X_B > 0)

Since ``X_A − X_B ~ N(μ_A − μ_B, sqrt(σ_A² + σ_B²))``::

    P(A > B) = Φ((μ_A − μ_B) / sqrt(σ_A² + σ_B²))

where Φ is the standard normal CDF (``scipy.stats.norm.cdf``).

This is the same model used in Bayesian bandit literature for online ad
optimisation (Chapelle & Li 2011, "An Empirical Evaluation of Thompson Sampling",
NeurIPS).

Effect size (Cohen's d)
~~~~~~~~~~~~~~~~~~~~~~~
::

    d = (μ_A − μ_B) / pooled_std
    pooled_std = sqrt((σ_A² + σ_B²) / 2)

    |d| < 0.2: negligible
    0.2 ≤ |d| < 0.5: small
    0.5 ≤ |d| < 0.8: medium
    |d| ≥ 0.8: large

Thompson sampling
~~~~~~~~~~~~~~~~~
For *n_samples* draws:

    1. Sample s_i ~ N(μ_i, σ_i) independently for each creative i.
    2. Record which creative had the largest sample (the "winner").

Win rate_i = fraction of draws where creative i won.  This equals P(A > B) for
two creatives, providing an independent Monte Carlo validation of the closed-form
result; agreement within ±1% is a strong sanity check.

Usage
-----
    from src.ab_testing import CreativeABTester

    tester = CreativeABTester(win_threshold=0.80)

    # Single pairwise comparison
    result = tester.compare_pair(ranked[0], ranked[1])
    print(f"P(A>B)={result.win_probability:.3f}, rec={result.recommendation}")

    # Thompson sampling — select top-3 from 10 candidates
    top3_indices = tester.thompson_select(ranked[:10], n_select=3)

    # Save all pairwise comparisons for a top-5 ranking
    all_results = tester.compare_all_pairs(ranked[:5])
    tester.save_results(all_results, "outputs/ab_results.json")

    # Visualisations
    tester.plot_comparison(result, ranked[0], ranked[1],
                           output_path="outputs/ab_comparison.png")
    tester.plot_tournament_heatmap(ranked[:5],
                                   output_path="outputs/ab_tournament.png")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# The 95% CI corresponds to ±1.96 standard deviations of a Gaussian.
_Z_95: float = 1.96

# Default P(A > B) threshold for "prefer_a" recommendation.
# 0.80 is deliberately conservative — we require at least 80% confidence
# before recommending A over B.  For high-stakes decisions use 0.90 or 0.95.
_WIN_THRESHOLD_DEFAULT: float = 0.80


@dataclass
class ABTestResult:
    """
    Statistical summary of a pairwise creative comparison.

    Fields
    ------
    creative_a_frame_idx : int
        Row index of creative A in the embeddings / index arrays.
    creative_b_frame_idx : int
        Row index of creative B.
    creative_a_video : str
        Source video identifier for creative A.
    creative_b_video : str
        Source video identifier for creative B.
    win_probability : float
        P(A > B) under the Gaussian CI model.  Range [0, 1].
        Values near 0.5 indicate indistinguishable performance.
    ci_overlap_fraction : float
        Fraction of the *shorter* 95% CI that overlaps with the other CI.
        0.0 = no overlap (decisive separation); 1.0 = complete overlap.
    cohens_d : float
        Standardised effect size.  Positive means A is better; negative
        means B is better.
    recommendation : str
        ``"prefer_a"`` | ``"prefer_b"`` | ``"inconclusive"`` based on
        the ``win_probability`` vs the configured ``win_threshold``.
    n_thompson_wins_a : int
        Thompson-sampling wins for A out of ``n_thompson`` draws.
    n_thompson_wins_b : int
        Thompson-sampling wins for B out of ``n_thompson`` draws.
    """

    creative_a_frame_idx: int
    creative_b_frame_idx: int
    creative_a_video: str
    creative_b_video: str
    win_probability: float
    ci_overlap_fraction: float
    cohens_d: float
    recommendation: str
    n_thompson_wins_a: int
    n_thompson_wins_b: int


class CreativeABTester:
    """
    Bayesian A/B testing and Thompson sampling for creative performance pairs.

    Args:
        win_threshold:  P(A > B) above which "prefer_a" is issued.
                        Symmetric: P(A > B) < 1 − win_threshold → "prefer_b".
                        Must be in (0.5, 1.0).  Default 0.80.
        n_thompson:     Thompson-sampling draws per creative.  Default 10 000
                        (less than 1% Monte Carlo error for most comparisons).
        random_state:   NumPy random seed for reproducibility.
    """

    def __init__(
        self,
        win_threshold: float = _WIN_THRESHOLD_DEFAULT,
        n_thompson: int = 10_000,
        random_state: int = 42,
    ) -> None:
        if not (0.5 < win_threshold < 1.0):
            raise ValueError(
                f"win_threshold must be in (0.5, 1.0); got {win_threshold}."
            )
        if n_thompson < 100:
            raise ValueError(
                # At n_thompson = 100 the Monte Carlo standard error for a
                # Bernoulli proportion is at most 1/(2*sqrt(100)) = 5%.
                # Fewer samples give unreliably noisy win-rate estimates.
                f"n_thompson must be ≥ 100 for reliable estimates "
                f"(SE ≤ 5 %%); got {n_thompson}."
            )
        self.win_threshold = win_threshold
        self.n_thompson = n_thompson
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_sigma(creative) -> Tuple[float, float]:
        """
        Extract ``(mu, sigma)`` from a :class:`src.ranking.RankedCreative`.

        ``sigma`` is derived from the 95% CI width under the normal
        approximation::

            sigma = (ci_upper - ci_lower) / (2 * 1.96)

        A floor of ``1e-6`` prevents degenerate zero-variance distributions
        when all bootstrap samples collapsed to a single value.
        """
        mu = float(creative.predicted_score)
        sigma = max(
            float(creative.ci_upper - creative.ci_lower) / (2.0 * _Z_95),
            1e-6,
        )
        return mu, sigma

    @staticmethod
    def _ci_overlap_fraction(
        lo_a: float, hi_a: float,
        lo_b: float, hi_b: float,
    ) -> float:
        """
        Overlap coefficient of two intervals, normalised by the shorter width.

        Returns 0.0 if the intervals do not overlap; 1.0 if one fully
        contains the other.
        """
        overlap = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
        width_a = max(hi_a - lo_a, 1e-10)
        width_b = max(hi_b - lo_b, 1e-10)
        return float(overlap / min(width_a, width_b))

    # ------------------------------------------------------------------
    # Core pairwise comparison
    # ------------------------------------------------------------------

    def compare_pair(self, creative_a, creative_b) -> ABTestResult:
        """
        Closed-form Bayesian comparison of two creatives.

        Computes P(A > B) analytically, Cohen's d effect size, CI overlap,
        and a recommendation.  Also runs Thompson sampling as an independent
        Monte Carlo validation.

        Args:
            creative_a:  A :class:`src.ranking.RankedCreative` instance.
            creative_b:  A :class:`src.ranking.RankedCreative` instance.

        Returns:
            :class:`ABTestResult` with all comparison statistics populated.
        """
        mu_a, sigma_a = self._score_sigma(creative_a)
        mu_b, sigma_b = self._score_sigma(creative_b)

        # Closed-form: P(A > B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B²))
        pooled_std = float(np.sqrt(sigma_a ** 2 + sigma_b ** 2))
        z = (mu_a - mu_b) / max(pooled_std, 1e-10)
        win_prob = float(norm.cdf(z))

        # Cohen's d effect size
        pooled_std_d = float(np.sqrt((sigma_a ** 2 + sigma_b ** 2) / 2.0))
        cohens_d = float((mu_a - mu_b) / max(pooled_std_d, 1e-10))

        # CI overlap fraction
        overlap = self._ci_overlap_fraction(
            float(creative_a.ci_lower), float(creative_a.ci_upper),
            float(creative_b.ci_lower), float(creative_b.ci_upper),
        )

        # Thompson sampling: draw n_thompson samples from each posterior
        samples_a = self.rng.normal(mu_a, sigma_a, size=self.n_thompson)
        samples_b = self.rng.normal(mu_b, sigma_b, size=self.n_thompson)
        wins_a = int((samples_a > samples_b).sum())
        wins_b = self.n_thompson - wins_a

        # Recommendation
        if win_prob > self.win_threshold:
            rec = "prefer_a"
        elif win_prob < 1.0 - self.win_threshold:
            rec = "prefer_b"
        else:
            rec = "inconclusive"

        logger.info(
            "A/B: %s(%.3f±%.3f) vs %s(%.3f±%.3f) → "
            "P(A>B)=%.3f  d=%.2f  overlap=%.2f  rec=%s.",
            creative_a.video_id, mu_a, sigma_a,
            creative_b.video_id, mu_b, sigma_b,
            win_prob, cohens_d, overlap, rec,
        )
        return ABTestResult(
            creative_a_frame_idx=int(creative_a.frame_idx),
            creative_b_frame_idx=int(creative_b.frame_idx),
            creative_a_video=str(creative_a.video_id),
            creative_b_video=str(creative_b.video_id),
            win_probability=round(win_prob, 6),
            ci_overlap_fraction=round(overlap, 6),
            cohens_d=round(cohens_d, 6),
            recommendation=rec,
            n_thompson_wins_a=wins_a,
            n_thompson_wins_b=wins_b,
        )

    # ------------------------------------------------------------------
    # Multiple comparisons
    # ------------------------------------------------------------------

    def compare_all_pairs(
        self,
        ranking: list,
        top_n: Optional[int] = None,
    ) -> List[ABTestResult]:
        """
        Run pairwise A/B tests for all C(N, 2) pairs in *ranking*.

        Args:
            ranking:  Output of :class:`src.ranking.CreativeRanker.rank`.
            top_n:    If provided, only compare the top *top_n* creatives.

        Returns:
            List of :class:`ABTestResult` sorted by decisiveness
            (``abs(win_probability − 0.5)`` descending — most decisive first).
        """
        if not ranking:
            return []
        creatives = ranking[:top_n] if top_n else ranking
        results: List[ABTestResult] = []
        for i in range(len(creatives)):
            for j in range(i + 1, len(creatives)):
                results.append(self.compare_pair(creatives[i], creatives[j]))
        results.sort(
            key=lambda r: abs(r.win_probability - 0.5), reverse=True
        )
        logger.info(
            "compare_all_pairs: %d pairs tested (%d creatives).",
            len(results), len(creatives),
        )
        return results

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def thompson_select(
        self,
        ranking: list,
        n_select: int,
        n_samples: int = 10_000,
    ) -> List[int]:
        """
        Select the top-*n_select* creatives by Thompson sampling win rate.

        For each of *n_samples* draws:

        1. Sample one score from ``N(μ_i, σ_i)`` for every creative ``i``.
        2. The creative with the highest sample wins that draw.

        The creative with the most wins is the most likely true best performer.

        This multi-armed bandit approach:

        - Handles > 2 creatives without requiring C(N,2) pairwise tests.
        - Trades off exploration (high uncertainty → more exploration) and
          exploitation (high mean → more selection).
        - Validates the closed-form P(A > B) for two creatives: the Thompson
          win rate should match P(A > B) to within ~1% at 10 000 samples.

        Args:
            ranking:    Output of :class:`src.ranking.CreativeRanker.rank`.
            n_select:   Number of top creatives to return.
            n_samples:  Number of Thompson draws.  Default 10 000.

        Returns:
            List of *n_select* 0-based indices into *ranking*, sorted by
            descending win rate.

        Raises:
            ValueError: if *n_select* > len(*ranking*).
        """
        if not ranking:
            return []
        n = len(ranking)
        n_select = min(n_select, n)

        mus    = np.array([c.predicted_score for c in ranking], dtype=np.float64)
        sigmas = np.array(
            [max((c.ci_upper - c.ci_lower) / (2.0 * _Z_95), 1e-6) for c in ranking],
            dtype=np.float64,
        )

        # (n_samples, n) matrix of draws; argmax along axis=1 = winner per draw
        draws   = self.rng.normal(mus, sigmas, size=(n_samples, n))
        winners = np.argmax(draws, axis=1)                        # (n_samples,)
        win_counts = np.bincount(winners, minlength=n)            # (n,)

        # Sort by descending win count and return top n_select indices
        selected = np.argsort(win_counts)[::-1][:n_select].tolist()

        logger.info(
            "Thompson select: %d creatives, %d draws → top %d: %s "
            "(win rates: %s).",
            n, n_samples, n_select, selected,
            [f"{win_counts[i] / n_samples:.2f}" for i in selected],
        )
        return selected

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: List[ABTestResult],
        output_path: str,
    ) -> str:
        """
        Save A/B test results to a JSON file.

        Args:
            results:      Output of :meth:`compare_all_pairs`.
            output_path:  Destination file path.

        Returns:
            Absolute path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = [asdict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        logger.info(
            "A/B test results saved to %s (%d comparisons).", output_path, len(rows)
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        result: ABTestResult,
        creative_a,
        creative_b,
        output_path: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (11, 5),
    ) -> str:
        """
        Two-panel comparison chart for a single A/B result.

        **Left panel**: score-distribution curves (Gaussian from CI model)
        for A (blue) and B (orange), with the fill region where A > B shaded.

        **Right panel**: Thompson sampling win-count bar chart (draws / n_thompson).

        Args:
            result:       Output of :meth:`compare_pair`.
            creative_a:   The :class:`src.ranking.RankedCreative` for A.
            creative_b:   The :class:`src.ranking.RankedCreative` for B.
            output_path:  Destination PNG file.
            title:        Optional figure-level title.
            figsize:      ``(width, height)`` in inches.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        mu_a, sigma_a = self._score_sigma(creative_a)
        mu_b, sigma_b = self._score_sigma(creative_b)

        # Build x-axis range covering 4σ around each mean
        lo = min(mu_a - 4 * sigma_a, mu_b - 4 * sigma_b)
        hi = max(mu_a + 4 * sigma_a, mu_b + 4 * sigma_b)
        xs = np.linspace(lo, hi, 600)

        pdf_a = norm.pdf(xs, mu_a, sigma_a)
        pdf_b = norm.pdf(xs, mu_b, sigma_b)

        fig, (ax_dist, ax_bar) = plt.subplots(1, 2, figsize=figsize)

        # --- Distribution panel ---
        ax_dist.plot(
            xs, pdf_a, color="#5486e0", linewidth=2,
            label=f"A: {creative_a.video_id} (μ={mu_a:.3f})",
        )
        ax_dist.plot(
            xs, pdf_b, color="#e07b54", linewidth=2,
            label=f"B: {creative_b.video_id} (μ={mu_b:.3f})",
        )
        # Fill A's distribution for intuitive "A wins" region
        ax_dist.fill_between(
            xs, pdf_a, alpha=0.18, color="#5486e0",
            label=f"P(A>B) = {result.win_probability:.2f}",
        )
        ax_dist.fill_between(xs, pdf_b, alpha=0.12, color="#e07b54")

        # Mean lines
        ax_dist.axvline(
            mu_a, color="#5486e0", linestyle="--", linewidth=1.2, alpha=0.8
        )
        ax_dist.axvline(
            mu_b, color="#e07b54", linestyle="--", linewidth=1.2, alpha=0.8
        )

        rec_colour = {
            "prefer_a": "#5486e0",
            "prefer_b": "#e07b54",
            "inconclusive": "#666666",
        }.get(result.recommendation, "black")

        ax_dist.set_title(
            f"Score distributions\n"
            f"P(A>B)={result.win_probability:.2f}  "
            f"Cohen's d={result.cohens_d:+.2f}  "
            f"→ {result.recommendation.replace('_', ' ').upper()}",
            fontsize=10, color=rec_colour,
        )
        ax_dist.set_xlabel("Predicted CTR score", fontsize=9)
        ax_dist.set_ylabel("Probability density", fontsize=9)
        ax_dist.legend(fontsize=8, loc="upper right")
        ax_dist.grid(linestyle=":", alpha=0.35)

        # --- Thompson sampling bar ---
        labels = [
            f"A\n{creative_a.video_id[:10]}",
            f"B\n{creative_b.video_id[:10]}",
        ]
        wins = [result.n_thompson_wins_a, result.n_thompson_wins_b]
        bar_colours = ["#5486e0", "#e07b54"]
        ax_bar.bar(labels, wins, color=bar_colours, alpha=0.82, edgecolor="white")
        ax_bar.set_ylabel(f"Thompson wins / {self.n_thompson:,}", fontsize=9)
        ax_bar.set_title("Thompson sampling win counts", fontsize=10)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.35)
        total = max(sum(wins), 1)
        for bar_i, w in enumerate(wins):
            ax_bar.text(
                bar_i, w + total * 0.01, f"{100 * w / total:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        if title:
            fig.suptitle(title, fontsize=11, y=1.02)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("A/B comparison chart saved to %s.", output_path)
        return os.path.abspath(output_path)

    def plot_tournament_heatmap(
        self,
        ranking: list,
        output_path: str,
        title: str = "P(row > col) — Creative Tournament",
        figsize: Tuple[int, int] = (9, 7),
        top_n: Optional[int] = None,
    ) -> str:
        """
        N×N heatmap of P(row creative beats col creative) for all pairs.

        Green (> 0.5) means the row creative is more likely to win;
        red (< 0.5) means the column creative wins.  The diagonal is 0.5
        by convention.

        This is a standard tournament matrix used in multi-armed bandit
        and preference learning literature to visualise pairwise dominance.

        Args:
            ranking:     Output of :class:`src.ranking.CreativeRanker.rank`.
            output_path: Destination PNG file.
            title:       Figure title.
            figsize:     ``(width, height)`` in inches.
            top_n:       Show only the top *top_n* creatives.

        Returns:
            Absolute path to the saved PNG.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        creatives = ranking[:top_n] if top_n else ranking
        n = len(creatives)

        if n < 2:
            logger.warning(
                "Tournament heatmap requires ≥ 2 creatives; skipping (n=%d).", n
            )
            return output_path

        # Build N×N win-probability matrix analytically
        matrix = np.full((n, n), 0.5, dtype=np.float32)
        for i in range(n):
            mu_i, sigma_i = self._score_sigma(creatives[i])
            for j in range(n):
                if i == j:
                    continue
                mu_j, sigma_j = self._score_sigma(creatives[j])
                pooled = float(np.sqrt(sigma_i ** 2 + sigma_j ** 2))
                z = (mu_i - mu_j) / max(pooled, 1e-10)
                matrix[i, j] = float(norm.cdf(z))

        tick_labels = [
            f"#{c.rank} {c.video_id[:8]}@{c.timestamp:.0f}s"
            for c in creatives
        ]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="P(row > col)")

        ax.set_xticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(title, fontsize=11, pad=10)

        # Annotate each cell with the P-value
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                text_colour = "white" if (val < 0.25 or val > 0.75) else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color=text_colour,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            "Tournament heatmap saved to %s (%d×%d matrix).", output_path, n, n
        )
        return os.path.abspath(output_path)
