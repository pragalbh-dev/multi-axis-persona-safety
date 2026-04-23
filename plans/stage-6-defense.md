# Stage 6: Defense Experiment (Exp 6)

**Objective:** Build and evaluate multi-axis activation capping as an improved defense. Show the Pareto frontier for PC1-only vs multi-axis capping.

**Prerequisites:** Stage 4 complete (know which PCs are exploitable and at what severity). Stage 5 results used as adversarial test cases if available.

**Completion criteria:** Multi-axis capping implemented and calibrated, Pareto frontiers compared, optimal defense dimensionality determined. Report section 8 drafted. Viz 4 built.

---

## Required inputs

- `progress.md` — **Stage 4 → Stage 6 Handoff**: exploitable PCs per model, adversarial null-space direction paths, layer ranges for PC2/PC3. Plus **Stage 5 → Stage 6 Handoff** if available: safety-concerning compositions.
- `CONVENTIONS.md` — validation/test split rule, Pareto analysis conventions.
- Cached role vectors + PCA from Stage 3, adversarial directions from Stage 4.

**Last task of this stage (after T6.6): append Stage 6 → Stage 7 Handoff (or Stage 6 → Stage 8 if skipping extensions) to `progress.md`. Must include: optimal multi-axis capping config per model (PCs + thresholds + layers), headline number (incremental harm reduction over PC1-only), adversarial robustness of the defense.**

---

## Tasks

- [ ] T6.1: Implement multi-axis capping
  - Extend the single-axis capping formula to N simultaneous PCs
  - For each PC i: `h ← h - v_i * min(⟨h,v_i⟩ - τ_i, 0)` applied sequentially
  - Configurations to test:
    - PC1 only (baseline — reproduce paper result)
    - PC1 + PC2
    - PC1 + PC2 + PC3
    - PC1 + all safety-relevant PCs (from Exp 2 LASSO selection)
  - Verify: capping order doesn't matter (directions are orthogonal, so it shouldn't)

- [ ] T6.2: Calibrate thresholds per PC
  - For each PC independently: sweep τ ∈ {1st, 10th, 25th, 50th, 75th percentile}
  - For multi-axis: also test cross-combinations (e.g., PC1 at 25th + PC2 at 10th)
  - Use held-out validation set (split 1,100 prompts: 550 val / 550 test)
  - Select optimal τ per PC on validation set

- [ ] T6.3: Full evaluation of each defense configuration
  - For each (capping config × model): run full safety eval (550 test prompts) + full capability eval
  - Also test against: Stage 4's adversarial null-space directions (does multi-axis capping block them?)
  - Also test against: Stage 5's safety-concerning compositions
  - Save full output tuples for Viz 6

- [ ] T6.4: Pareto frontier analysis
  - For each model: plot harm reduction vs capability for every capping configuration
  - Compare: PC1-only frontier vs multi-axis frontier
  - Compute: incremental harm reduction per added PC (diminishing returns curve)
  - Report: "optimal defense dimensionality" — how many PCs do you need to cap?

- [ ] T6.5: Build Viz 4 (Layer-by-Layer Persona Signal)
  - Layer slider: sweep through model layers
  - At each layer: show PCA projection, variance explained, PC stability
  - Line chart: persona signal strength per layer
  - Add to dashboard

- [ ] T6.6: Draft report section 8
  - Section 8: Multi-Axis Defense — configurations tested, calibration, Pareto frontiers, optimal dimensionality, adversarial robustness
  - Key framing: "Multi-axis capping improves the Pareto frontier by X% at zero capability cost" or "PC1-only capping is sufficient — additional axes provide negligible benefit"
  - Include all figures
  - Add to `report/paper.md`

---

## Expected Outputs

- `results/exp6_multi_axis_defense/` — per-config harm rates, capability scores, Pareto frontiers
- Viz 4 in dashboard
- Report section 8 drafted
- **Key deliverable:** the recommendation to labs — "cap these N PCs at these thresholds for optimal safety"

---

## Notes

- If multi-axis is only marginally better (< 5% additional harm reduction): the paper's PC1-only approach is nearly optimal. Still report the finding — it saves other labs from over-engineering.
- If multi-axis is substantially better (> 15% additional): this is the practical contribution. Labs should upgrade.
- The validation/test split is important — don't calibrate and evaluate on the same data. Use the first 550 prompts for threshold selection, the last 550 for final numbers.
- Capping on orthogonal directions should be order-independent. Verify this empirically — if there's an order effect, the PCs may not be truly orthogonal in the capping layer (they're orthogonal in the middle-layer PCA space, but capping happens at different layers).
