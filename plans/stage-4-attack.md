# Stage 4: Attack Experiments (Exp 3 + Exp 4)

**Objective:** Test whether PC1 capping can be bypassed via orthogonal steering (Exp 3) and construct worst-case adversarial directions in PC1's null space (Exp 4). This is the core safety-critical finding.

**Prerequisites:** Stage 3 complete (know which PCs are safety-relevant, have logistic-LASSO model).

**Subjects:** Tier 1 (3 models) + Gemma 4 31B dense (both modes) = 5 subjects, same as Stage 3.

**Completion criteria:** For each subject: steering curves for safety-relevant PCs, random baselines scaled to lmsys-chat-1m norm, capping bypass rates, adversarial null-space results (LASSO-based only). Report sections 5-6 drafted. Viz 3 built.

---

## Required inputs

- `progress.md` — **Stage 3 → Stage 4 Handoff**: per-model list of safety-relevant PCs, LASSO model path (for T4.6 adversarial direction), baseline harm rates.
- `CONVENTIONS.md` — random baseline rule (always 5 matched-norm random vectors; do not skip), seed rule, parquet row format for steering sweeps.
- Cached role vectors + PCA from Stage 3 (don't recompute).

Can run in parallel with Stage 5 (no cross-dependency). If running in parallel, coordinate judge server usage — two stages hitting the same judge will fight for GPU time.

**Last task of this stage (after T4.8): append Stage 4 → Stage 6 Handoff to `progress.md`. Must include: capping bypass rate per model/PC, adversarial direction paths (for Stage 6 defense testing), layer-sweep results for PC2/PC3.**

---

## Experiment 3: Orthogonal Steering

### Tasks

- [ ] T4.0: Calibrate Assistant Axis (AA) capping config per subject (paper's 2D sweep; prerequisite for T4.5, Stage 5 T5.5, Stage 6 T6.1)
  - **Direction = AA, not PC1** per CONVENTIONS "Primary intervention direction" + paper §3.1 line ~468. Read AA direction per subject from `configs/assistant_axis.yaml` (Stage 3 T3.1.5 output). Paper's capping results in Figure 10 are AA-based; PC1-based capping is an Appendix G.3.2 comparison only.
  - Paper's capping layer range is NOT tied to the extraction layer (paper line 691): chosen by an **independent 2D sweep** (center × width) × τ percentile, Pareto-optimizing harm-reduction × capability-preservation. Capping centers in the paper fall in the middle-to-late depth band (deeper than the extraction middle).
  - **Tier 1 path — use paper's published AA optima verbatim:**
    - Read `configs/paper_capping_ranges.yaml` pulled in Stage 0 T0.7. Qwen 3 32B: layers 46-53 (center 49.5, width 8), τ = 25th percentile. Llama 3.3 70B: layers 56-71 (center 63.5, width 16), τ = 25th percentile. Gemma 2 27B: read from the file (paper reports in Appendix F).
    - Compute τ per subject = 25th percentile of role-rollout **AA projections** at the capping layers (multi-layer range, NOT the extraction layer) — use Stage 3 T3.1 per-layer mean-response-token activation cache + per-layer AA direction.
    - **Reproduction sanity check:** run AA capping on the full 1,100-prompt jailbreak set for each Tier 1 subject. Verify harm-rate reduction is in the ballpark of paper's ~60% (Figure 10). If off by >15pp in either direction, debug (τ distribution, AA direction, hook placement, layer indexing) — paper used AA too, so this is a direct reproduction test. Log any divergence to `decisions.md`.
  - **Tier 2 path — run the 2D sweep ourselves (no published numbers):**
    - For each Tier 2 subject (Gemma 4 31B thinking ON + OFF): sweep centers ∈ {40%, 50%, 60%, 70%, 80% of layer depth} × widths ∈ {4, 8, 16, 24} × τ percentiles ∈ {1st, 10th, 25th, 50th, 75th} on AA. For each configuration: evaluate on a 200-prompt stratified jailbreak subsample + the 4-benchmark capability subsample cached in Stage 3 T3.5.5 (same subsample per subject for Pareto comparability).
    - Pick the (center, width, τ) triple on the Pareto frontier with ≥50% harm reduction at minimal capability loss (paper's Figure 10 selection rule). If no config achieves ≥50%, pick best harm reduction on the frontier and document gap in `decisions.md`.
  - **Output:** `configs/aa_capping.yaml` per subject with `{model_id, direction: "assistant_axis", capping_layer_range: [L_start, L_end], capping_center, capping_width, tau_percentile, tau_value, reproduction_harm_reduction, source: "paper_verbatim" | "pareto_sweep"}`. Stage 4 T4.5, Stage 5 T5.5, Stage 6 T6.1 all consume this — do not recompute τ or layer range elsewhere.
  - **Compute budget — Tier 2 sweep:** 5 centers × 4 widths × 5 percentiles × 200 safety + 1,000 capability problems per config × 2 Tier 2 subjects ≈ 120K runs. Batched on 4×5090 ≈ 6-8 hours per Tier 2 subject.

- [ ] T4.1: Implement per-PC steering experiments
  - **PC selection:** the **primary** safety-relevant set from Stage 3 T3.8 (LASSO-nonzero in binary logistic model), plus PC2 and PC3 regardless. Read `results/exp2_safety_relevance/safety_relevant_pcs_<model>.yaml`.
  - **Per-model per-PC reporting (mandatory — do NOT pool blindly):** read `configs/pc_pooling.yaml` from Stage 3 T3.3. For each PC, report steering curves per-subject. Only average curves across subjects whose entry in the pooling table has that PC in the same pool group (cos_sim > 0.7). If PC2 is pooled for (Qwen 3, Llama 3.3) but stands alone for (Gemma 2, Gemma 4 ON, Gemma 4 OFF), produce one pooled curve for the first pair and separate per-model curves for the rest. Do not compute a "mean PC2 effect across all subjects" that averages non-aligned subspaces.
  - For each (subject, PC) under evaluation:
    - 9 strengths: λ ∈ {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    - **Sample budget (compute cut):** run the **full 1,100 prompts only at λ=0 (baseline) and at the strongest-effect strength** (determined after a quick λ∈{-2, +2} pilot on 200 prompts). For the remaining 7 strengths, use a **stratified 500-prompt subsample** of the 1,100 (stratified across Shah et al.'s 44 harm categories, ~11 prompts/category). Saves ~55% of generations while preserving curve shape.
    - Capability eval: run **only at λ ∈ {-2, -1, 0, +1, +2}** (5 strengths, not 9). Capability doesn't need the fine-grained curve.
  - Steering scale: λ multiplies the unit direction vector × the average post-MLP residual stream norm on lmsys-chat-1m at the extraction layer (paper's convention, cached by Stage 0 T0.6).
  - Save full output tuples for Viz 6.
  - Config: `configs/exp3.yaml` specifying PCs, strengths, layers, subjects, sample sizes per strength.

- [ ] T4.2: Run random vector baselines
  - **Norm-matched construction:** 5 random unit vectors per target PC, each scaled to the **average post-MLP residual stream norm on lmsys-chat-1m at the extraction layer** (matches paper line 474, cached in `data/cache/lmsys_norms/`).
  - Run the same steering sweep (9 strengths, same sample-size budget as T4.1).
  - Compare: does the target PC produce MORE harm than random vectors at matched strength?
  - This is the critical control — without it, any "finding" could be generic perturbation.

- [ ] T4.3: Layer sweep for PC2/PC3
  - The paper found optimal capping at layers 46-53 (Qwen) and 56-71 (Llama) for PC1
  - Test: is the optimal layer for PC2/PC3 steering the same?
  - Sweep: 50%, 60%, 70%, 80%, 90% depth for each model × PC2/PC3
  - Use 200 prompts (subsample) for efficiency
  - Report: optimal layer ranges per PC per model

- [ ] T4.4: Analyze steering curves
  - Plot: harm rate vs λ for each PC (one curve per PC, one panel per model)
  - Look for: monotonicity (linear effect) vs phase transitions (manifold evidence)
  - Overlay random baseline band (mean ± 1 SD across 5 random vectors)
  - Compute: statistical significance of PC effect vs random at each strength

---

## Experiment 4: Blind Spot Construction

### Tasks

- [ ] T4.5: Test AA capping + orthogonal PC2/PC3 steering (direct H1 test)
  - Apply **AA capping** at the per-subject config from T4.0 (`configs/aa_capping.yaml`) — this is the paper's actual defense, reproduced.
  - THEN steer on PC2/PC3 (or the primary LASSO-selected PCs from Stage 3 T3.8) at various strengths on the AA-capped model. PC2+ are orthogonal to PC1 and ≈orthogonal to AA since cos_sim(PC1, AA) > 0.71 — so AA capping shouldn't intercept these steers.
  - Measure: does harm increase despite AA being capped?
  - **Blind spot severity = ASR(AA capped + PC_i steered) − ASR(AA capped, no steering).** Positive gap = AA capping has a blind spot along PC_i. This is the direct empirical test of H1.

- [ ] T4.6: Construct adversarial null-space direction (LASSO-based, AA-orthogonal)
  - **Method (locked, paper-aligned — direction is in AA's null space, not just PC1's):** from Stage 3 T3.8's fitted logistic-LASSO model, take coefficients `c_i` on PCs 2..k (LASSO was already fit with AA as the baseline feature, so PC1 is not in the feature set). Build `u_adv_raw = Σ_{i≥2} c_i · v_i` where `v_i` is the i-th PC direction at L*. PCs 2..k are orthogonal to PC1 by construction but NOT strictly orthogonal to AA (since PC1 ≠ AA exactly).
  - **Explicit AA-projection step:** `u_adv = u_adv_raw − (⟨u_adv_raw, AA⟩ / ||AA||²) · AA`. This guarantees `⟨u_adv, AA⟩ = 0` — the direction lives strictly in AA's null space, so AA capping cannot intercept steering along it.
  - L2-normalize `u_adv`, scale to lmsys-chat-1m reference norm at L*.
  - Steer on `u_adv` at 5 strengths λ ∈ {-2, -1, 0, +1, +2} on the AA-capped model (use per-subject config from T4.0).
  - Evaluate safety (stratified 500-prompt subsample for intermediate strengths, full 1,100 at λ=±2).
  - **Method 2 (projected gradient) is cut.** The differentiable-proxy options reduce to a linear probe on PCs → harm, mathematically equivalent to the LASSO coefficients we already have.

- [ ] T4.7: Build Viz 3 (Steering Effect Comparator)
  - Dropdown: select PC to steer
  - Slider: steering strength
  - Display: activation distribution shift, harm rate, capability scores
  - Random baseline overlay
  - Add to dashboard

- [ ] T4.8: Draft report sections 5-6
  - Section 5: Orthogonal Steering — steering curves, random baselines, layer sweep, statistical significance
  - Section 6: Blind Spot Construction — capping bypass rates, adversarial directions, severity quantification
  - Include all figures
  - Add to `report/paper.md`

---

## Expected Outputs

- `results/exp3_orthogonal_steering/` — per-condition harm rates, capability scores, full output tuples
- `results/exp4_blind_spots/` — capping bypass rates, adversarial directions, severity metrics
- Viz 3 in dashboard
- Report sections 5-6 drafted
- **Key number:** the blind spot severity = how much more harm can be induced while PC1 capping is active

---

## Notes

- If capping bypass rate is near zero: PC1 capping is sufficient even against orthogonal attacks. H1 is rejected. Still a contribution.
- If capping bypass rate is significant (>10% increase in ASR): this is the headline finding. Labs using PC1 capping are deploying an incomplete defense.
- The projected gradient method (T4.6, Method 2) requires the judge to be differentiable or a differentiable proxy. If using an LLM judge, use Method 1 (LASSO-based) as primary and Method 2 as a stretch goal with a learned proxy.
- Stage 4 and Stage 5 are independent — can run in parallel if GPU bandwidth allows. Stage 4 is more important (core safety claim), so prioritize it if resources are limited.
