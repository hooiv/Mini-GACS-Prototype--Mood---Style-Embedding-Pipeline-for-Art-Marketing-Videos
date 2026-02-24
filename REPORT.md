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
                              similarity matrix (N×N cosine)
                                      │
                    ┌─────────────────┼────────────────────┐
                    ▼                 ▼                    ▼
               heatmap PNG      top-k grids        Markdown report
```

---

## 2. Extending into a Production GACS Engine

### 2a. Affective Score Taxonomy

CLIP embeddings can be enriched with targeted **text-guided probing**:

```python
# Example: score a frame on an "energetic vs. serene" axis
prompts = ["an energetic, dynamic scene", "a calm, minimalist scene"]
text_embs = model.encode_text(prompts)         # (2, D)
frame_emb  = model.encode_image(frame)         # (1, D)
axis_score = (frame_emb @ text_embs.T).squeeze()  # shape (2,)
```

This maps each frame to a named affective axis without any labelled training
data, replacing the unsupervised cosine distance with an interpretable score
(e.g. *energy*, *warmth*, *complexity*, *luxury*).

### 2b. Video-Level Vibe Aggregation

Frame-level scores can be time-pooled (mean, max, or attention-weighted) to
produce a **video-level vibe vector**.  That vector can then be stored in a
vector database (e.g. Qdrant, Pinecone) and used to cluster a large creative
library by mood — the foundation of a GACS-grade creative intelligence engine.

### 2c. Multi-Modal Extension

Adding **audio embeddings** (e.g. CLAP model) and **transcript sentiment**
(BERT-based) alongside visual embeddings enables a fused affective score that
is more robust to edge cases where visual cues are ambiguous.

---

## 3. Connecting to Performance Feedback (CTR / ROAS)

### Next Step A — Vibe–Performance Regression

Collect historical ad creative assets together with their CTR or ROAS
figures.  Compute vibe embeddings for each creative.  Train a lightweight
regression head (e.g. ridge regression or a 2-layer MLP) that maps
*vibe embedding → performance metric*.  This model can then be used to
**score new creatives before launch**, surfacing the predicted CTR of a
draft video within seconds.

```
Vibe embedding ──► Ridge / MLP head ──► predicted CTR / ROAS
```

Validation strategy:
- Hold out a time-stratified 20 % of campaigns for evaluation.
- Track Spearman ρ between predicted and actual CTR.
- A/B test: route a fraction of new creatives through the scorer and check
  whether selected creatives outperform randomly chosen ones.

### Next Step B — Retrieval-Augmented Creative Optimisation

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
| Test scaffolding | ChatGPT-4o for test case ideas | Reviewed each proposed test for correctness; removed two that were tautological |
| Matplotlib heatmap | Copilot template | Ran locally; inspected PNG; adjusted `vmin/vmax` and tight-layout manually |

**Key verification practices used throughout:**

1. **Shape assertions** — every embedding array is checked for expected `(N, D)` shape.
2. **NaN / Inf guards** — explicit checks after every model call and after loading from disk.
3. **Round-trip tests** — metadata and embeddings are saved then reloaded and compared.
4. **Determinism test** — identical input images must produce identical embeddings (within float32 tolerance).
5. **Mock-based unit tests** — the CLIP model is mocked so tests run without internet or GPU.

---

## 5. Limitations and Future Work

- **Frame sampling is uniform**: scene-cut detection (e.g. PySceneDetect) would
  produce more semantically representative frames.
- **Single-modal embeddings**: audio and transcript signals are not yet included.
- **No temporal ordering**: the similarity matrix treats all frames as a bag;
  a temporal graph (frames as nodes, edges weighted by similarity) would capture
  narrative arc.
- **CLIP is not fine-tuned on advertising data**: domain-adaptive fine-tuning on
  a labelled creative dataset would improve affective sensitivity.
