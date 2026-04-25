# Multi-Axis Persona Safety: Geometry of Persona Space Beyond PC1

> Stage 1 wireframe. Each section names the experiment whose `metrics.json`
> fills it and the figure(s) it carries. Replace placeholder bullets with
> prose as Stages 3-6 produce results.

---

## Abstract

<!-- DATA NEEDED: aggregate H1-H4 results across results/exp{1..6}/metrics.json -->
<!-- FIGURES: Fig 2 (teaser) -->

3-5 sentence summary. State the headline numbers from H1 (blind-spot lift),
H3 (orthogonal-attack ASR), and H4 (multi-axis defense Pareto).

## 1. Introduction

<!-- DATA NEEDED: none — written from scope doc -->

- Motivate persona-axis safety: paper's PC1 capping reduces harm by ~60%
  (Lu et al., line 691 in extracted text), but persona space has more
  structure. Does multi-axis matter?
- Contribution preview (4 hypotheses):
  - **H1.** Higher PCs (PC2, ..., PCk) carry independent safety signal beyond PC1.
  - **H2.** Persona-space dimensionality is small (≪ d_model).
  - **H3.** Orthogonal-PC steering can attack a model whose PC1 is capped.
  - **H4.** Multi-axis capping improves the safety/capability Pareto over PC1-only.
- Threat model: open-weight, activation-level (per `Non-Surjective` paper, NOT
  prompt-level).

## 2. Background & Related Work

<!-- DATA NEEDED: none — literature -->

- Lu et al. 2601.10387 (the paper we're extending). Key results we replicate:
  cos_sim(PC1, AA) > 0.71 at the middle layer; PC1 capping at the deeper
  range (Qwen 46-53, Llama 56-71) ≈60% harm reduction; cross-model PC1
  stability across 3 Tier 1 subjects.
- Shah et al. 2311.03348 (persona vectors / steering).
- Non-Surjective paper 2604.09839 (steered states have no prompt pre-image —
  bounds our threat-model claims).
- Statistical methods: Marchenko-Pastur threshold, BCa bootstrap, BH-FDR,
  logistic LASSO.

## 3. Persona Space Decomposition (Exp 1)

<!-- DATA NEEDED: results/exp1_pca_decomposition/metrics.json -->
<!-- FIGURES: Fig 1 (Persona Space 3D), Fig 4 (Layer-by-Layer Persona Signal) -->

- Eigenspectrum per subject (4 subjects after Stage 0 → 1 revert: Gemma 2 27B,
  Qwen 3 32B, Gemma 4 31B thinking-ON, Gemma 4 31B thinking-OFF). Stage 7 Ext 9
  adds Llama 3.3 70B at fp8 if GPU budget permits.
- MP threshold γ = d/n per subject (n = role-vector count after fully/somewhat
  split, expected 300-500 per CONVENTIONS line 98).
- PC interpretation: AA ≈ PC1 validation (per-subject cos_sim(PC1, AA) > 0.7;
  log to decisions.md if any subject falls below).
- Effective dimensionality: top-k cumulative variance for k explaining ≥70%.

## 4. Safety Relevance of Higher PCs (Exp 2)

<!-- DATA NEEDED: results/exp2_safety_relevance/metrics.json -->
<!-- FIGURES: Fig 2 (Safety Heatmap Across PCs) -->

- Per-PC point-biserial against binarized harm; BH-FDR at q=0.05.
- Aggregate Pearson across (continuous projection × continuous harm rate)
  per condition — paper-style r=0.39-0.52 framing.
- **Headline: blind-spot lift** = `AUC(AA + LASSO-selected PCs 2..k) − AUC(AA only)`,
  bootstrap BCa CI on the delta. Lift > 0 supports H1.
- Cohen's d for harmful-vs-non-harmful projection differences (medium ≥ 0.5,
  large ≥ 0.8).
- Secondary ordinal LASSO check (3-level collapse). Reported alongside if
  it disagrees with the primary binary fit.

## 5. Orthogonal Steering (Exp 3)

<!-- DATA NEEDED: results/exp3_orthogonal_steering/metrics.json -->
<!-- FIGURES: Fig 3 (Steering Effect Comparator) -->

- ASR and capability deltas under PCi steering (i ∈ {2, 3, ...}) with the
  baseline λ ∈ [-2, 2] step 0.5. 5 random unit-vector baselines per PC,
  each scaled to lmsys-chat-1m residual-stream norm at the extraction layer.
- Steering applied at the single extraction layer (paper §3.2.1 line 474),
  positions=all.
- Pooling rule: cross-model average only when pairwise cos_sim > 0.7
  (`configs/pc_pooling.yaml` from Stage 3 T3.3).

## 6. Blind Spot Construction (Exp 4)

<!-- DATA NEEDED: results/exp4_blind_spot/metrics.json -->
<!-- FIGURES: Fig 3 (Steering Comparator showing AA-capped + adversarial-PCi attack curves) -->

- Adversarial direction `u_adv = Σ c_i · PC_i` with explicit AA-projection
  removal (CONVENTIONS line 113). Capping fact baseline: AA at paper's
  capping range (verbatim for Tier 1; 2D-swept for Tier 2 in Stage 4 T4.0).
- Blind-spot severity = `ASR(AA capped + u_adv steered) − ASR(AA capped only)`.
- Stage 4 cuts: 500-prompt stratified subsample at 7 intermediate λ; full
  1,100 only at λ=0 and the strongest attack.

## 7. Persona Composition (Exp 5)

<!-- DATA NEEDED: results/exp5_composition/metrics.json -->
<!-- FIGURES: Fig 5 (Persona Arithmetic) -->

- Linearity test: `α=β=0.5` locked for primary fit (per progress.md
  2026-04-24 17:00). Variable-α,β fit deferred to Stage 7 Ext 8.
- Test: do compound personas (`role_A + role_B`) live near the linear
  combination in d_model space? Manifold evidence if they don't.

## 8. Multi-Axis Defense (Exp 6)

<!-- DATA NEEDED: results/exp6_multi_axis_defense/metrics.json -->
<!-- FIGURES: Fig 6 (Multi-Axis Defense Pareto) -->

- Configurations: `{AA}`, `{AA + PC2}`, `{AA + PC2 + PC3}`,
  `{AA + all LASSO-selected PCs}`. Phase A defaults to per-direction τ
  inheritance from Stage 4 T4.0; Phase B sweeps cross-percentile only when
  Phase A shows ≥10% additive multi-axis gain.
- Orthogonality check at every capping layer (Stage 6 T6.1); if cos_sim > 0.1
  anywhere, fall back to deterministic capping order PC1→PC2→PC3 and log to
  decisions.md.
- Pareto: harm rate × capability score under each config. Headline = lift in
  harm reduction at matched capability cost vs paper's PC1-only baseline.

## 9. Discussion

<!-- DATA NEEDED: synthesis across exps 1-6 + Stage 7 extensions -->

- Implications for jailbreak robustness: multi-axis ≠ free-lunch unless
  blind-spot lift > random-baseline noise.
- Cross-model differences: Gemma 4 thinking ON vs OFF is the project's
  novel comparison axis (paper does no reasoning models).
- Limitations: 4 subjects (vs paper's 3 + extensions); Llama 70B cited from
  paper; threat model is activation-level only.
- Future work pointers (Stage 7 extensions list).

## 10. Conclusion

<!-- DATA NEEDED: 1-paragraph wrap of headline numbers -->

3-4 sentences. Restate H1-H4 outcomes with headline numbers.

---

## Appendices

- **A. Reproducibility:** subject configs, judge prompt, eval-size table,
  seeds. Pointer to `plans/CONVENTIONS.md` "Scientific conventions" section.
- **B. Per-subject AA × PC1 cos_sim and extraction-layer choices** (Stage 3
  T3.1.5 output).
- **C. Capping-range sweep tables** (Stage 4 T4.0).
- **D. Judge validation:** 200-sample GPT-5.5 pseudo-ground-truth agreement
  numbers (Stage 2 T2.4.5).
