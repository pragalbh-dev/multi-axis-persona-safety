# Stage 3: Foundation Experiments (Exp 1 + Exp 2)

**Objective:** Establish what persona space looks like (Exp 1) and whether it contains safety-relevant information beyond PC1 (Exp 2). These produce the foundation all subsequent experiments build on.

**Prerequisites:** Stage 2 complete (infrastructure smoke test passes).

**Completion criteria:** For each Tier 1 model: full eigenspectrum with MP threshold, PC interpretation table, per-PC safety correlations with FDR correction, blind spot fraction quantified. Report sections 3-4 drafted. Viz 1-2 built.

---

## Required inputs

- `progress.md` — **Stage 2 → Stage 3 Handoff**: module interfaces, judge endpoint URL + latency/throughput, activation cache parquet schema, config template path.
- `CONVENTIONS.md` — seed rule, parquet schema, judge prompt template (filled in Stage 2), checkpointing principle.
- Pre-computed role vectors from the assistant-axis HuggingFace repo (downloaded in Stage 0).
- Tier 1 models loaded and verified (thinking OFF on Qwen 3).

**Last task of this stage (after T3.10): append Stage 3 → Stage 4 Handoff to `progress.md`. Must include: list of safety-relevant PCs per model (for Stage 4 steering direction selection), LASSO model path (for Stage 4 adversarial direction construction), blind spot fraction per model.**

---

## Experiment 1: Persona Space Decomposition

### Tasks

- [ ] T3.1: Load pre-computed persona role vectors
  - Download from assistant-axis HuggingFace for all 3 Tier 1 models
  - Verify: 275 archetypes, correct dimensionality, matches paper description
  - If any are missing or corrupted: extract from scratch using assistant-axis pipeline

- [ ] T3.2: Run PCA with full eigenspectrum analysis
  - For each model: center role vectors, run PCA
  - Compute and report: variance explained per PC (PC1, PC2, ..., PCk) — this is the gap in the original paper
  - Apply Marchenko-Pastur threshold: compute γ = d/n ≈ 4096/275, upper edge λ+ = σ²(1+√γ)²
  - Report: effective dimensionality (how many PCs above MP threshold)
  - Generate: scree plot with MP threshold line

- [ ] T3.3: Cross-model PC stability analysis
  - Compute cosine similarity between same-numbered PCs across all 3 models (3 pairs)
  - Verify: PC1 stability > 0.92 (matches paper)
  - Report: full cross-model similarity matrix for PCs 1-10
  - Flag: which PCs are stable (>0.7) and which are model-specific (<0.5)

- [ ] T3.4: PC interpretation via loading analysis
  - For each PC (1 through k): rank all 275 roles by projection
  - Top 10 and bottom 10 roles per PC — what semantic axis does this suggest?
  - Pre-annotate roles with attributes (authority, formality, harm-propensity, human-vs-nonhuman, fictional-vs-real) and compute correlation with each PC
  - Generate: role loading heatmap per PC

- [ ] T3.5: Build Viz 1 (Persona Space Explorer)
  - 3D scatter: all roles projected onto PC1/PC2/PC3
  - Color by category, hover shows role name + projections + harm rate (from Exp 2)
  - Toggle between models
  - Add to dashboard

---

## Experiment 2: Safety Relevance of Higher PCs

### Tasks

- [ ] T3.6: Run baseline safety evaluation
  - 1,100 jailbreak prompts × 3 models, NO intervention (just default model behavior)
  - Extract activations at evaluation layer for each prompt-response pair
  - Save: full (prompt, response, activations, safety_score) tuples
  - Report: baseline harm rates per model (paper reports 65.3%-88.5%)

- [ ] T3.7: Compute per-PC safety correlations
  - For each prompt: project activations onto each PC (1 through k)
  - For each PC: Spearman correlation between projection and harm score
  - Apply BH-FDR correction across all PCs
  - Compute Cohen's d for harmful vs safe response projections per PC
  - Report: table of (PC, correlation, p-value, FDR-adjusted p, Cohen's d)

- [ ] T3.8: LASSO joint safety prediction
  - Features: projections onto all k PCs
  - Target: harm score (binary or ordinal)
  - Nested 10-fold CV for unbiased R²
  - Report: which PCs survive LASSO selection, joint R², comparison to PC1-only R²
  - **The key metric: blind spot fraction = 1 - R²(PC1 only)**

- [ ] T3.9: Build Viz 2 (Safety Heatmap)
  - Matrix: PC index (columns) × harm category (rows, from the 44 categories in Shah et al.)
  - Cell color = correlation strength
  - FDR-significant cells highlighted
  - Add to dashboard

- [ ] T3.10: Draft report sections 3-4
  - Section 3: Persona Space Decomposition — eigenspectrum, dimensionality, PC interpretations, cross-model stability
  - Section 4: Safety Relevance — per-PC correlations, LASSO results, blind spot fraction
  - Include all figures generated in this stage
  - Add to `report/paper.md`

---

## Expected Outputs

- `results/exp1_pca_decomposition/` — eigenvalues, PC directions, role loadings, cross-model similarities
- `results/exp2_safety_relevance/` — per-prompt safety scores, PC projections, correlation tables, LASSO model
- Viz 1 and Viz 2 in dashboard
- Report sections 3-4 drafted
- **Key number to carry forward:** which PCs are safety-relevant (used in Stage 4 to choose steering directions)

---

## Notes

- If blind spot fraction < 5%: PC1 is nearly sufficient. H1 is likely false. Still a contribution (validation), but pivot focus to composition (Stage 5).
- If blind spot fraction > 20%: strong evidence for multi-dimensional safety. H1 and H2 are likely true. Stage 4 becomes very exciting.
- The pre-annotated role attributes (T3.4) can be generated with an LLM. Ask Claude or GPT-4 to label each of the 275 archetypes on a 1-5 scale for each attribute. Quick and cheap.
- Save the LASSO model — we'll use its coefficients to construct the "most dangerous blind spot direction" in Stage 4.
