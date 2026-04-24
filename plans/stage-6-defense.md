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
  - Extend the single-axis capping formula to N simultaneous directions
  - For each direction `d` in the config: `h ← h − d · min(⟨h, d⟩ − τ_d, 0)` applied sequentially at every layer in the capping range and at every token position (per CONVENTIONS)
  - **Configurations to test** (baseline is Assistant Axis, NOT PC1 — per CONVENTIONS + paper §3.1 line ~468):
    - **AA only** (baseline — reproduces paper's result; read τ + layer range from `configs/aa_capping.yaml` written by Stage 4 T4.0)
    - AA + PC2
    - AA + PC2 + PC3
    - AA + all primary safety-relevant PCs (LASSO-nonzero set from Stage 3 T3.8)
  - **Verify AA/PC near-orthogonality at every capping layer (NOT just the extraction center):** AA is defined per-layer; PCs are computed at L* only. At each layer L in the capping range, compute pairwise cos_sim of `{AA_L, PC2_L, PC3_L, ...}` using the per-layer AA cache (T3.1) + per-layer PC directions (T3.1.5). Expect AA ≈ PC1 direction at L* (cos_sim > 0.71), so AA should have low cos_sim with PC2+ (bounded below ~0.7 by orthogonality of PCs). If any pair of DISTINCT axes in the capping config (e.g., AA × PC2) exceeds 0.1 at any capping layer: order-independence fails → switch to deterministic **fixed order = AA → PC2 → PC3 → rest of LASSO set** and document. Record the per-layer `{axis_pair: cos_sim}` matrix to `results/exp6_multi_axis_defense/orthogonality_check.json`.

- [ ] T6.2: Calibrate thresholds + layer range per PC (2-phase, conditional second phase)
  - **Per-direction (capping layer range, τ) calibration — paper's 2D sweep per added axis:**
    - **AA uses the already-calibrated config from Stage 4 T4.0** (`configs/aa_capping.yaml`). Do not redo.
    - For each additional PC (PC2, PC3, and any LASSO-selected PCs from Stage 3 T3.8): paper did not test multi-axis capping, so we calibrate each PC's (layer range, τ) ourselves.
      - **Tier 1:** as a compute-saving default, assume each additional PC's optimal capping layer range matches AA's (paper's Qwen 46-53, Llama 56-71, Gemma 2 from Appendix F); sweep τ only. If Phase A shows the per-PC optimal τ fails to add Pareto gain, fall back to a Tier-2-style 2D sweep for that PC as a debug path.
      - **Tier 2 (Gemma 4 31B both modes):** run the same (center × width × τ percentile) grid as Stage 4 T4.0 for each added PC on a 200-prompt safety × capability subsample. Centers ∈ {40%, 50%, 60%, 70%, 80% depth}, widths ∈ {4, 8, 16, 24}, τ percentiles ∈ {1st, 10th, 25th, 50th, 75th}.
    - **Training data:** held-out 550-prompt validation split of the 1,100 (other 550 reserved for test).
  - **Phase A (always run) — 4 multi-axis configurations:** `{AA}`, `{AA+PC2}`, `{AA+PC2+PC3}`, `{AA + all primary safety-relevant PCs from Stage 3 T3.8 LASSO}`. Each direction uses its own optimal (layer_range, τ) from the calibration above.
  - **Phase B (conditional, only if Phase A shows ≥10% additive gain):** full cross-percentile sweep over τ (keep layer ranges locked at per-direction optima). 5 percentiles × k axes = 5^k configs — worth running only if Phase A demonstrates meaningful multi-axis benefit. Cap at 3 axes (AA + 2 PCs → 5^3 = 125 configs) even if Phase A shows more PCs are safety-relevant.
  - **Decision rule for Phase B:** proceed only if `ASR(AA+PC2) ≤ 0.9 × ASR(AA only)` on validation. Otherwise stop at Phase A and report "per-direction optimal percentile is sufficient; cross-percentile tuning gives no further gain."
  - Select final config on validation set; evaluate on held-out test set.

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
