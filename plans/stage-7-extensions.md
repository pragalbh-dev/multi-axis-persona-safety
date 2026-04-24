# Stage 7: Extensions

**Objective:** Run ranked extensions that address additional paper limitations, each following the same implement → run → analyze → report pattern.

**Prerequisites:** Stage 6 complete (core results in hand).

**Completion criteria:** Each completed extension has results, analysis, and a report section/appendix.

---

## Required inputs

- `progress.md` — **Stage 6 → Stage 7 Handoff**: finalized multi-axis defense config (for Ext 1-2 to test on Tier 2), all cached artifacts paths, open questions raised by core results.
- `CONVENTIONS.md` — all sections, especially the model IDs (to avoid re-deciding Tier 2 IDs).
- Tier 2 models loaded and verified in Stage 0.

**Reasoning-model handling (Ext 2) is already partially subsumed by Tier 2's dual-mode runs (thinking ON/OFF on Gemma 4 31B and Qwen 3.6-35B-A3B MoE).** Ext 2 here means: deeper analysis of thinking vs non-thinking differences in persona space — not a separate model load. Update T7.4–T7.6 accordingly when picking this up.

**Last task of each extension: append a Stage 7.{ext} Handoff block to `progress.md` so later extensions or Stage 8 can find the artifacts.**

---

## Extension priority (from scope doc, revised 2026-04-24)

Gemma 4 31B dense (both thinking modes) was **promoted into Stages 3/4/6** as a core subject. What remains in Extensions is the MoE case and deeper reasoning analysis that doesn't fit the core run.

| Rank | Extension | Effort | Value | Status |
|------|-----------|--------|-------|--------|
| 1 | Qwen 3.6-35B-A3B MoE — full pipeline | Medium-High | Medium | pending |
| 2 | Reasoning deep-dive (thinking vs answer subspace geometry) | Low | Medium-High | pending |
| 3 | Improved persona elicitation | Medium | High | pending |
| 4 | Decorrelated persona subset | Low | Medium | pending |
| 5 | Per-expert MoE router analysis | Medium-High | Medium | pending |
| 6 | SAE feature mapping | Medium | Medium | pending |
| 7 | Trait-persona decomposition | Low | Medium | pending |
| 8 | Variable α, β composition fit (revisit Stage 5) | Low | Low | pending |

---

## Ext 1: MoE (Qwen 3.6-35B-A3B) — full pipeline

**Why extension and not core:** Gemma 4 31B dense is in core stages. MoE needs nnsight-backed hooks for clean residual-stream extraction across the expert-routing layer, plus optional per-expert analysis (Ext 5). The tooling work gates the run.

### Tasks

- [ ] T7.1: MoE extraction harness
  - Verify nnsight can hook the post-routing residual stream on Qwen 3.6-35B-A3B. Residual stream after MoE block aggregates experts into one vector — standard extraction works here.
  - Decide: also cache per-expert contribution (gate logit × expert output) for Ext 5? If yes, schema change in `data/cache/activations/` (add `expert_id` dimension).
  - Document in `configs/model_hooks.yaml`.

- [ ] T7.2: Extract persona vectors for MoE
  - 275 roles × **300 rollouts each** (reduced from paper's 1,200 — same cut we applied in Stage 3 for Gemma 4 dense). Pipeline is the same as assistant-axis extraction, filter via primary judge.
  - Thinking mode ON + OFF (same as Gemma 4 dense), extract thinking and answer tokens separately.
  - Save to `data/cache/activations/qwen3.6-35b-a3b-moe/`.

- [ ] T7.3: Run Exps 1-6 on MoE
  - Reuse all experiment scripts with the MoE subject config.
  - Compare to Tier 1 + Gemma 4 dense from core stages: does MoE persona space differ? Is the Assistant Axis still PC1? Does capping transfer?
  - Add MoE results as a dedicated section in the report (paper's §8.1 explicitly names MoE as a gap — this is a first-class contribution).

---

## Ext 2: Reasoning deep-dive (thinking vs answer subspace geometry)

**Why extension:** Core Stage 3 already extracts activations from thinking and answer tokens separately for Gemma 4 31B dense in thinking-ON mode. That gives two PCA spaces. Ext 2 is the **deep comparison** between them — the geometry questions beyond "is the Assistant Axis still PC1 in both?"

### Tasks

- [ ] T7.4: Subspace alignment analysis
  - Compute Grassmann / principal angles between the top-k subspaces of thinking-token PCA and answer-token PCA on Gemma 4 31B thinking-ON.
  - Question: is the persona subspace the same geometric object when the model is thinking vs answering?

- [ ] T7.5: Drift of Assistant Axis across thinking tokens
  - Project activations at every thinking token (not just mean) onto the Assistant Axis.
  - Does the projection wander during thinking and re-anchor at the answer, or stay stable?
  - If it wanders: implication for reasoning-model safety — the thinking phase may cross persona boundaries that the answer doesn't expose.

- [ ] T7.6: Safety correlation in thinking space
  - Run Stage 3 Exp 2 equivalent using thinking-token activations as input features.
  - Are the same PCs safety-relevant, or does thinking carry distinct safety signal?
  - Blind spot lift computed separately for thinking vs answer subspaces.

---

## Ext 3: Improved Persona Elicitation

### Tasks

- [ ] T7.7: Generate augmented persona set
  - ~50 adversarial personas mined from Shah et al. and HarmBench jailbreak prompts
  - ~30 psychometric personas from Big Five × professional role combinations
  - ~20 multi-turn emergent personas (longer conversation prompts)

- [ ] T7.8: Extract augmented persona vectors
  - Run extraction pipeline on augmented set
  - Combine with original 275 for expanded PCA

- [ ] T7.9: Compare expanded PCA
  - Does the eigenspectrum change? New PCs emerge?
  - Do adversarial personas cluster in specific (previously undersampled) PCs?
  - Report: what the original 275 missed

---

## Ext 4: Decorrelated Persona Subset

### Tasks

- [ ] T7.10: Select decorrelated subset
  - From 275 personas, greedily select ~100 that maximize minimum pairwise angular distance
  - Algorithm: start with the two most distant, iteratively add the persona most distant from all selected

- [ ] T7.11: Compare PCA quality
  - Run PCA on decorrelated subset
  - Compare: are PCs more interpretable? Is variance more evenly distributed?
  - Report: methodological recommendation for future persona space studies

---

## Ext 5: MoE-Specific Analysis (on Qwen 3.6)

### Tasks

- [ ] T7.12: Analyze expert routing patterns
  - For each persona: which experts fire most often?
  - Cluster: do personas with similar PC projections use similar expert sets?
  - Question: does expert 47 encode safety? Does expert 12 encode persona X?

- [ ] T7.13: Stability analysis
  - Is persona PCA noisier in MoE vs dense? (compare Qwen 3 dense vs Qwen 3.6 MoE)
  - Does the router carry persona-relevant information beyond the residual stream?

---

## Ext 6: SAE Feature Mapping

### Tasks

- [ ] T7.14: Load pre-trained SAEs
  - Gemma 2 + Gemma Scope 2, or Llama 3.x + Goodfire SAEs
  - Verify: can extract SAE feature activations on our models

- [ ] T7.15: Map SAE features to PCs
  - For each PC: which SAE features correlate most strongly?
  - For safety-relevant PCs: are the corresponding SAE features interpretable? (e.g., "deception," "authority," "compliance")
  - Report: from "PC2 is safety-relevant" to "PC2 is safety-relevant because features X, Y, Z live there"

---

## Ext 7: Trait-Persona Decomposition

### Tasks

- [ ] T7.16: Extract trait vectors
  - Use PERSONA paper's methodology (Big Five contrastive activation analysis) on one Tier 1 model
  - Or download from their repo if available: github.com/xcfcode/persona

- [ ] T7.17: Project personas onto trait subspace
  - For each of 275 personas: project onto the 5 (or 10) trait directions
  - Compute reconstruction error: ||v_persona - projection||
  - Report: are personas well-explained by traits, or is there significant residual?

---

## Ext 8: Variable α, β composition fit (revisit Stage 5)

### Tasks

- [ ] T7.18: Per-pair α, β fit
  - For each persona pair from Stage 5: fit (α, β) ∈ [0,1]² minimizing L2 residual between predicted and empirical composition.
  - Compare: fitted-α,β residuals vs locked-0.5,0.5 residuals. Is the linearity story preserved under free coefficients, or do complementary pairs need asymmetric weights?
  - Low priority — mostly a sensitivity check on Stage 5's primary result.

---

## Notes

- Extensions are strictly ordered by priority. Don't start Ext 2 before Ext 1 is done (unless GPU-idle time makes it efficient).
- Each extension should add a section or appendix to the report, not modify existing core sections.
- Time constraint: fellowship deadline May 3, 2026. Extensions beyond Ext 2 are likely post-deadline.
