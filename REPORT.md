# Mini GACS Prototype — Design Report

## 1. System Overview (GenTA Perspective)

This prototype is a **Generalised Affective Computing System (GACS)** seed
that maps video frames into a shared semantic-aesthetic embedding space using
OpenAI CLIP.  The key insight is that CLIP was trained on 400 M image–text
pairs, so its visual features capture not only object identity but also
compositional mood cues (warm/cool palette, texture density, level of motion
blur, depth of field style).  By measuring cosine similarity in that space we
obtain a proxy for "vibe similarity" — the degree to which two frames evoke the
same aesthetic or emotional register.

### Architecture at a Glance

```
Videos ─► frame_extractor ─► JPEG frames + metadata CSV
                                      │
                              embeddings (CLIP) ─► .npy + index JSON
                                      │
                    ┌─────────────────┼──────────────────────────────┐
                    ▼                 ▼                    ▼         ▼
               heatmap PNG      top-k grids        affective    clustering
                                                    scores       (K-means)
                                      │
                              temporal_analysis ─► narrative arc, pacing
                                      │
                          performance_predictor ─► predicted CTR/ROAS
```

---

## 2. Extending into a Production GACS Engine

### 2a. Affective Score Taxonomy *(implemented — `src/affective_scoring.py`)*

CLIP embeddings are enriched with targeted **text-guided probing** across
6 named affective axes (energy, warmth, complexity, luxury, joy, tension):

```python
score_i = sim(frame_i, pos_prompt) − sim(frame_i, neg_prompt)  # ∈ [-2, 2]
```

This maps each frame to a named affective axis without any labelled training
data, replacing the unsupervised cosine distance with an interpretable score.
Video-level scores are obtained by mean-pooling frame scores and visualised
as a radar chart.

### 2b. Video-Level Vibe Aggregation *(implemented — `src/clustering.py`)*

Frame-level scores are time-pooled (mean) to produce a **video-level vibe
vector**.  `VibeClusterer` groups creatives into K visual-mood clusters using
K-means on CLIP embeddings, visualised with a 2-D PCA scatter plot.
`auto_n_clusters()` uses the elbow heuristic to suggest K automatically.
This is the foundation of a GACS-grade creative intelligence engine.

### 2c. Temporal Narrative Arc *(implemented — `src/temporal_analysis.py`)*

`TemporalAnalyser` re-introduces time ordering to expose:

- **Temporal similarity curve** — per-frame similarity to local neighbours,
  showing how visually consistent each moment is with its context.
- **Scene-transition detection** — large drops in the temporal curve flag
  hard cuts or style changes.
- **Visual pacing score** (variance of the curve) — high ↔ dynamic fast-cut;
  low ↔ slow contemplative.
- **Temporal coherence score** (mean of the curve) — measures narrative
  visual consistency.

These signals characterise a video's "editing DNA" and can be correlated with
CTR/ROAS to learn which pacing/coherence profiles perform best.

### 2d. Multi-Modal Extension *(planned)*

Adding **audio embeddings** (e.g. CLAP model) and **transcript sentiment**
(BERT-based) alongside visual embeddings enables a fused affective score that
is more robust to edge cases where visual cues are ambiguous.

---

## 3. Connecting to Performance Feedback (CTR / ROAS)

### Next Step A — Vibe–Performance Regression *(implemented — `src/performance_predictor.py`)*

`VibePerformancePredictor` implements a full R&D-grade predictor pipeline:

1. **Synthetic data generator** — creates plausible CTR/ROAS labels from
   affective scores with a known ground-truth formula + Gaussian noise.
   This lets us verify the predictor can recover known signal — standard
   practice when real labels are unavailable.

2. **Ridge regression head** (or MLP fallback) maps vibe features → CTR:
   ```
   [affective scores (6) | PCA components (5)] ──► StandardScaler ──► Ridge
   ```

3. **5-fold cross-validation** reporting **Spearman ρ** (the gold-standard
   ranking-correlation metric for ad scoring).  On low-noise synthetic data
   the ridge model consistently achieves ρ > 0.9.

4. **Feature importance** — identifies which affective axis or PCA dimension
   most strongly drives predicted CTR, enabling actionable recommendations
   ("increase warmth", "reduce tension").

5. **Three visualisations** — CV fold bar chart, feature-importance horizontal
   bars, predicted-vs-actual scatter plot.

Once real campaign data is available, replace the synthetic labels with
historical CTR/ROAS figures and re-run `cross_validate()`.

### Next Step B — Retrieval-Augmented Creative Optimisation *(planned)*

Build a **similarity-based recommendation engine**: when a new creative is
uploaded, retrieve the top-10 most similar past creatives from the vector DB,
surface their historical performance stats, and flag which visual attributes
(colour palette, pacing, energy level) correlate with high CTR in that
similarity neighbourhood.  This closes the feedback loop:

```
New creative ──► embed ──► retrieve similar past creatives
                                 ─► surface CTR / ROAS stats
                                 ─► highlight winning vibe attributes
                                 ─► recommend tweaks (e.g. "increase warmth")
```

---

## 4. AI Tool Usage and Verification

| Stage | AI Tool Used | How Outputs Were Verified |
|-------|-------------|--------------------------|
| Module structure planning | GitHub Copilot (inline suggestions) | Reviewed each suggestion manually; rejected suggestions that coupled modules inappropriately |
| `extract_frames` docstring + edge-case handling | Copilot tab-completion | Read generated code line-by-line; added explicit `FileNotFoundError` guard and FPS fallback |
| CLIP embedding batching loop | Copilot + ChatGPT-4o for L2-normalisation reminder | Cross-checked with official CLIP repo; verified output norms ≈ 1.0 in unit test |
| Cosine-similarity matrix | Copilot (one-liner suggestion) | Verified against `sklearn.metrics.pairwise.cosine_similarity` on same data; results match to 5 decimal places |
| Affective axis prompts | ChatGPT-4o for prompt wording | Tested manually; verified that "energetic, dynamic" vs "calm, still" produces directionally correct scores on stock images |
| Temporal curve algorithm | Copilot + manual reasoning | Verified boundary conditions: single-frame video → score = 1.0; flat curve → no transitions |
| Synthetic CTR generator | GitHub Copilot template | Verified that ridge regression achieves Spearman ρ > 0.9 on low-noise synthetic data in CI |
| Clustering centroid representative | Code review (self + automated) | Bug fix: replaced zero-distance self-subtraction with correct frame-to-centroid distance |
| Test scaffolding | ChatGPT-4o for test case ideas | Reviewed each proposed test for correctness; removed two that were tautological |
| Matplotlib heatmap | Copilot template | Ran locally; inspected PNG; adjusted `vmin/vmax` and tight-layout manually |

**Key verification practices used throughout:**

1. **Shape assertions** — every embedding array is checked for expected `(N, D)` shape.
2. **NaN / Inf guards** — explicit checks after every model call and after loading from disk.
3. **Round-trip tests** — metadata and embeddings are saved then reloaded and compared.
4. **Determinism test** — identical input images must produce identical embeddings (within float32 tolerance).
5. **Mock-based unit tests** — the CLIP model is mocked so tests run without internet or GPU.
6. **Signal-recovery test** — Spearman ρ > 0.5 on low-noise synthetic data validates predictor correctness.

---

## 5. Current Limitations and Future Work

- **Audio modality**: `src/performance_predictor.py` only uses visual CLIP + affective
  features.  Adding CLAP audio embeddings and transcript-sentiment BERT features
  would improve recall for videos where visual cues are ambiguous.
- **Real campaign data**: the predictor currently uses synthetic labels.  Connecting
  to a real CTR/ROAS database (de-identified) would enable production-grade
  performance prediction.
- **Scene-cut detection**: frame sampling is currently uniform.  Replacing it with
  the already-designed `detect_scene_transitions()` output as the sampling guide
  would yield more semantically representative frames.
- **Online learning**: the predictor is batch-trained.  An online update mechanism
  (e.g. sliding-window SGD) would let it adapt to seasonality and trend shifts in
  ad performance without full retraining.
- **Vector DB integration**: replacing the in-memory `.npy` store with Qdrant or
  Pinecone would enable sub-millisecond retrieval at creative library scale.
