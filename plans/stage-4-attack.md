# Stage 4: Attack Experiments (Exp 3 + Exp 4)

**Objective:** Test whether PC1 capping can be bypassed via orthogonal steering (Exp 3) and construct worst-case adversarial directions in PC1's null space (Exp 4). This is the core safety-critical finding.

**Prerequisites:** Stage 3 complete (know which PCs are safety-relevant, have LASSO model).

**Completion criteria:** For each Tier 1 model: steering curves for safety-relevant PCs, random baselines, capping bypass rates, adversarial null-space results. Report sections 5-6 drafted. Viz 3 built.

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

- [ ] T4.1: Implement per-PC steering experiments
  - For each safety-relevant PC identified in Exp 2 (plus PC2 and PC3 regardless):
    - Steer at 9 strengths: λ ∈ {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    - Run full safety eval (1,100 prompts) + capability eval at each strength
    - Save full output tuples for Viz 6
  - Config: `configs/exp3.yaml` specifying PCs, strengths, layers, models

- [ ] T4.2: Run random vector baselines
  - Generate 5 random vectors of matched norm for each safety-relevant PC
  - Run same steering sweep on each random vector
  - Compare: does the safety-relevant PC produce MORE harm than random vectors at matched strength?
  - This is the critical control — without it, any "finding" could be generic perturbation

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

- [ ] T4.5: Test PC1 capping + orthogonal steering
  - Apply PC1 capping at the paper's optimal settings (25th percentile, optimal layers)
  - THEN steer on PC2/PC3 at various strengths on the capped model
  - Measure: does harm increase despite PC1 being capped?
  - The gap: ASR(capped + PC2 steered) vs ASR(capped only) = blind spot severity

- [ ] T4.6: Construct adversarial null-space direction
  - Method 1 (LASSO-based): use LASSO coefficients from Exp 2 to build the linear combination of PCs 2-k that maximizes harm prediction, orthogonal to PC1
  - Method 2 (projected gradient): compute ∇ₕ(harm_score), project out PC1 component: `u_adv = (I - v₁v₁ᵀ) · ∇ₕ(harm_score)`
  - Steer on each adversarial direction at 5 strengths
  - Evaluate safety on capped model

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
