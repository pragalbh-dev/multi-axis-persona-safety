# Phase 5 — Writeup additions for the GitHub Pages page

This is a working draft of the new sections to be inserted into
`docs/index.html` after Phase 3 + Phase 4 outputs land. Numbers in
**[brackets]** are placeholders that get filled from:

- Phase 1: `results/plan_b_gemma2_27b/extensions/baseline_extended.parquet`
- Phase 2: `results/plan_b_gemma2_27b/extensions/harm_direction_merged.json`
- Phase 3: `results/plan_b_gemma2_27b/extensions/attack_arm/lambda_pareto.json`
- Phase 4: `results/plan_b_gemma2_27b/extensions/defence_arm/interaction_matrix.parquet`

---

## Section to insert: "Cross-dataset replication"

> Insert AFTER `Pipeline validation` h2, BEFORE `What this run found`.

The original page reported on n=500 DAN prompts. The post-deadline extension
ran the full DAN sample (n=1,100 net of the 2 length-outliers) plus the 1,105
Shah-reconstructed jailbreak prompts (Shah et al. 2311.03348 methodology;
synthetic minimal-perturbation persona priming).

| dataset | n | baseline harm rate (Wilson 95% CI) |
|---|---|---|
| DAN (in-the-wild jailbreaks, ~2,500 chars/prompt) | 1,102 | **14.7%** [12.7, 16.9] |
| Shah-reconstructed (synthetic, ~160 chars/prompt) | 1,105 | **7.3%** [5.9, 9.0] |

The single-subject finding is robust to dataset style: AA-cap defends both,
but harm-prediction geometry differs. **Shah's elaborate-jailbreak rate is
half of DAN's even though both are designed to elicit harm.** The shorter,
direct persona priming gives the model less prompt-time priming to anchor
itself in role-mode; AA-cap has less to defend against.

[FIGURE: cross_dataset_baseline.png]

## Section to insert: "Geometry of harm — v_harm decomposition"

> REPLACES the existing `Why look past PC1` section (which used scree-plot motivation).

The original blind-spot evidence (LASSO AUC delta +0.240) showed that
something orthogonal to AA carries harm-relevant information. To identify
*what*, we computed the DiffMean direction — the activation-space contrast
between harm-positive and harm-negative baseline cases:

> `v_harm = mean(harm-positive at L*=21) − mean(harm-negative at L*=21)`,
> on the merged DAN + Shah baseline (n=2,207, 243 harm-positive).

Decomposing v_harm onto the existing role-PCA basis:

| axis | cos(v_harm, axis) |
|---|---|
| **PC2** (sign-matched: −1) | **−0.638** ← argmax |
| PC3 (sign-matched: −1) | −0.428 |
| PC1 | −0.379 |
| AA | −0.356 |
| PC4 (sign-matched: +1) | +0.323 |
| residual outside top-10 PCs | 13.4% |

**Single-direction harm classifier AUC:**

| direction | AUC | 95% CI |
|---|---|---|
| ⟨h, v_harm⟩ | **0.89** | [0.86, 0.92] |
| ⟨h, −AA⟩ | 0.73 | (= 1 − 0.27 of the AA-positive direction) |

[FIGURE: v_harm_decomposition.png]

**Reading:**
- v_harm is **not** primarily AA. Only ~12% of v_harm's energy projects onto
  AA. The H1 hypothesis ("harm-relevant signal in directions orthogonal to
  AA") is supported geometrically.
- v_harm is **most strongly anti-aligned with PC2**. In our extraction's
  sign convention, positive PC2 points toward safety; the harm direction
  is approximately −PC2 plus minor contributions from PC3, PC4, AA.
- v_harm is a **substantially stronger single-direction harm classifier than
  AA alone** (0.89 vs 0.73). This captures most of the LASSO multi-direction
  AUC of 0.967 in a single direction.
- 86% of v_harm's energy lives in the top-10 role-PCA basis; the
  remaining residual outside is small. **Role-PCA is the right basis for
  harm geometry on Gemma 2 27B**; we do not need to pivot to harm-PCA.

## Section to insert: "Sign convention finding — and why the Plan B PC2 attack steered the wrong way"

> NEW section, AFTER "Geometry of harm".

PCA components have arbitrary sign by construction (PC2 vs −PC2 are equally
valid). In our extraction's sign convention, **PC2 happens to point toward
safety** — `cos(v_harm, PC2) = −0.64`. This is itself a finding: the
role-PCA basis on Gemma 2 27B has a deterministic geometric relationship
with the harm direction, where positive role-PC2 movement = away from harm.

The original Plan B page reported PC2 attacks at *positive* λ=0.25, observing
0% harm and 67% nonsense and framing the result as "PC attacks bypass the
cap's behavioural signature." Decomposing v_harm reveals that those positive-
λ PC2 attacks were geometrically steering **away from harm**, not toward it.
The 67% nonsense + 0% harm pattern was the model being pushed past its
Assistant register into off-distribution non-Assistant non-harmful states.

The H1-correct adversarial probe is to steer **toward v_harm** — equivalently,
PC2 at *negative* λ. We do this in the next section ("Per-axis attack arm")
with sign-matched directions: for each PC_i, attack vector = sign(cos(v_harm,
PC_i)) · PC_i.

## Section to insert: "Per-axis attack arm at coherence-preserving λ"

> REPLACES the original `Behavioural bypass` section.

Lu et al. (Appendix G.2) report coherence loss along AA and role-PC1 at
"sufficiently high steering strengths" (their Figures 4-5 sweep λ ∈ [-3, 3]).
Plan B at λ=0.25 already produced 67% nonsense for non-AA directions. We
swept finer and found the coherence-preserving operating regime is **λ ∈
[0.05, 0.15]** for non-AA axes — an order of magnitude smaller than the
paper's AA range, on the same Gemma 2 27B.

Per-axis adaptive sweep (mini-runs n=50 stratified, 25 harm-positive +
25 control), pick λ_max_coherent (largest λ with coherence ≥ 80%):

| attack axis | λ_max_coherent | mini harm rate | mini coherence |
|---|---|---|---|
| sign-matched PC2 | **0.05** | 10% | 92% |
| sign-matched PC3 | **0.05** | 10% | 90% |
| sign-matched PC4 | **0.15** | 0% | 90% |
| v_harm-direct | **0.10** | 12% | 80% |
| random_0 | 0.05 | 10% | 86% |
| random_1 | 0.05 | 14% | 82% |

**Full-sample harm rates (n=486 stratified, 243 harm-positive + 243 control):**

[Phase 3 full run results — PLACEHOLDERS]
- AA-cap baseline (no attack): [BASELINE_HARM]%
- AA-cap + PC2 sign-matched at λ=0.05: [PC2_HARM]%
- AA-cap + PC3 sign-matched at λ=0.05: [PC3_HARM]%
- AA-cap + PC4 sign-matched at λ=0.15: [PC4_HARM]%
- AA-cap + v_harm-direct at λ=0.10: [VHARM_HARM]%
- AA-cap + multi-PC composite: [MULTI_HARM]%

[FIGURE: lambda_pareto.png]

**Reading [pending full data]:**
- If single-axis attacks all sit near AA-cap baseline → **AA-cap is
  geometrically near-complete on this subject** at coherence-preserving λ.
  Refines the H1 finding: the LASSO orthogonal-PC signal is correlate-only,
  not causally exploitable at coherent steering magnitudes.
- If v_harm-direct or multi-PC composite recovers harm meaningfully → there
  is exploitable signal beyond AA, just below the coherence-loss threshold.

This experiment also extends Lu et al.'s Appendix G.2 caveat with a quantitative
finding: the coherence-preserving λ window is **axis-specific**, narrowing
sharply for higher-PC and learned harm directions. Their λ ∈ [-3, 3] guidance
applies to AA only.

## Section to insert: "Multi-axis defence — closing the residual blind spot"

> REPLACES "The defense and what it leaves behind" h2 OR sits AFTER it as a "follow-up" section.

[Phase 4 interaction matrix — PLACEHOLDERS]

We tested four cap configurations against the strongest Phase 3 attacks:

| defence config | axes capped | strongest-attack harm | refusal | coherence |
|---|---|---|---|---|
| D1: AA only (Plan B baseline) | {AA} | [...] | [...] | [...] |
| D2: AA + sign-matched PC2 | {AA, −PC2} | [...] | [...] | [...] |
| D3: AA + sign-matched PC2/PC3/PC4 | {AA, −PC2, −PC3, +PC4} | [...] | [...] | [...] |
| D4: AA + v_harm-direct | {AA, +v_harm} | [...] | [...] | [...] |

[FIGURE: interaction_matrix.png]

**Reading [pending]:**
- If D2/D3/D4 reduce the strongest-attack harm rate below D1 → **multi-axis
  defence is operationally better than single-axis AA-cap.** H2 supported.
- If they do not → AA-cap was already absorbing the linear harm signal in
  this subject; the LASSO-detected orthogonal signal is informative but
  not actionable as a defence target.
- If multi-axis caps break coherence on the 0-attack baseline → the per-axis
  τ calibration (currently p25) is too aggressive when stacked; needs to
  back off to p10 on the weaker axes.

## Section to update: "Headline numbers"

Replace existing table with:

| | value |
|---|---|
| Total prompts (DAN + Shah merged baseline) | 2,207 |
| Baseline harm rate (DAN) | 14.7% [12.7, 16.9] |
| Baseline harm rate (Shah) | 7.3% [5.9, 9.0] |
| AA-cap reduces DAN harm | 14.8% → 1.8% (Plan B) |
| Single-direction harm-AUC: ⟨h, v_harm⟩ | **0.89** [...] |
| Single-direction harm-AUC: ⟨h, −AA⟩ | 0.73 |
| LASSO AUC delta (Plan B): {AA + PC1..PC10} − {AA only} | +0.240 [+0.21, +0.27] |
| Argmax axis aligned with v_harm | PC2 (cos = −0.64) |
| Coherence-preserving λ (PC2/PC3) | 0.05 |
| Coherence-preserving λ (v_harm-direct) | 0.10 |
| Coherence-preserving λ (paper's AA) | ~2-3 (Lu et al. Fig 5) |
| Multi-axis defence: best config | [Phase 4 result] |
| Cross-judge agreement (GPT-5.5 vs Qwen 3.6) | 93% |

## Section to update: "Research questions for the next phase"

Strike out resolved:
- ~~"Predictive vs causal — does steering along the data-derived harm direction
  recover harm?"~~ → resolved by Phase 3 full run (see Per-axis attack arm)
- ~~"Geometry of harm — is v_harm orthogonal to AA?"~~ → resolved by Phase 2
- ~~"Multi-axis defence — does capping AA + harm-aligned PCs close the
  blind-spot signal?"~~ → resolved by Phase 4

Remaining (carry forward):
- Mechanistic interpretation of the role-PCs (especially PC2/PC3 individually)
- Cross-model: does the v_harm-aligns-with-PC2 finding hold on Qwen 3 32B
  and Gemma 4 31B? (post-deadline cross-subject sweep)
- Causal verification with prompt-level attacks (Non-Surjective constraint:
  steered states have no prompt pre-image; the harm direction defined
  here may not transfer to prompt-only access)

---

# Implementation checklist (Phase 5)

When Phase 4 outputs land:

1. Run figure generators:
   - `uv run python -m src.visualization.lambda_pareto`
   - `uv run python -m src.visualization.interaction_matrix`
2. Copy figures to `docs/figures/`:
   - `cp results/plan_b_gemma2_27b/extensions/figures/*.{png,html} docs/figures/`
3. Edit `docs/index.html` per the section additions above.
4. Test the page locally: `python3 -m http.server -d docs/ 8080` then visit.
5. `git add docs/ && git commit -m "Phase 5 extension writeup with cross-dataset, v_harm, attack-arm, multi-axis-defence"`
6. Push to GitHub for Pages auto-deploy.
