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

## Extension priority (from scope doc)

| Rank | Extension | Effort | Value | Status |
|------|-----------|--------|-------|--------|
| 1 | New models (Gemma 4, Qwen 3.6) | Medium | High | pending |
| 2 | Reasoning model analysis | Low-Medium | Medium-High | pending |
| 3 | Improved persona elicitation | Medium | High | pending |
| 4 | Decorrelated persona subset | Low | Medium | pending |
| 5 | MoE-specific analysis | Medium-High | Medium | pending |
| 6 | SAE feature mapping | Medium | Medium | pending |
| 7 | Trait-persona decomposition | Low | Medium | pending |

---

## Ext 1: New Models

### Tasks

- [ ] T7.1: Load Tier 2 models
  - Gemma 4 31B IT — verify loads via TransformerLens or nnsight
  - Qwen 3.6-35B-A3B — verify MoE model loads, identify hook points for residual stream
  - Document: VRAM, parallelism strategy, any compatibility issues

- [ ] T7.2: Extract persona vectors for Tier 2
  - Run the assistant-axis pipeline on Gemma 4 and Qwen 3.6 (no pre-computed axes exist)
  - 275 roles × 1,200 rollouts each = 330K rollouts per model
  - Save role vectors to `data/tier2_role_vectors/`

- [ ] T7.3: Run Exps 1-6 on Tier 2 models
  - Reuse all experiment scripts with new model configs
  - Compare results to Tier 1: do findings generalize to newer architectures?
  - Add Tier 2 results to report as extension section or appendix

---

## Ext 2: Reasoning Model Analysis

### Tasks

- [ ] T7.4: Set up reasoning ablation
  - Qwen 3 32B with thinking mode enabled vs disabled
  - Define: extract activations during thinking tokens, answer tokens, or both

- [ ] T7.5: Compare persona spaces
  - Run PCA on thinking-enabled and thinking-disabled separately
  - Compare: eigenspectrum, PC interpretations, cross-mode PC similarity
  - Question: does the thinking phase show different persona structure?

- [ ] T7.6: Safety eval on reasoning model
  - Run Exp 2-4 equivalent on thinking-enabled Qwen 3
  - Compare: are the same PCs safety-relevant? Is the blind spot fraction different?

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

## Notes

- Extensions are strictly ordered by priority. Don't start Ext 2 before Ext 1 is done (unless GPU-idle time makes it efficient).
- Each extension should add a section or appendix to the report, not modify existing core sections.
- Time constraint: fellowship deadline May 3. Extensions beyond Ext 2-3 are likely post-deadline.
