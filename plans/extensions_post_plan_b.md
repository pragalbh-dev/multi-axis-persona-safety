# Post Plan B — Research Extensions

**Status:** post-deadline plan. Plan B sign-fixed re-run is the working snapshot
(`results/plan_b_gemma2_27b/` as of 2026-04-26 05:09).
**Author/owner:** pragalbh-dev.
**Last updated:** 2026-04-26.

## Context

Plan B's sign-fixed re-run landed paper-class numbers:
- Baseline 14.8% → AA-cap 1.8% (Δ 13.0 pp, BCa 95% CI on the difference).
- LASSO blind-spot AUC delta = +0.240 [+0.209, +0.273], 8 of top-10 role PCs selected.
- Within-capped Cohen's d on PC2 = −1.68, PC3 = −1.39.
- Behavioural bypass: refusal-keyword incidence drops 76% → 17% → 2% under PC2/PC3
  attacks at λ=0.25 (raw denominator; coherence-adjusted: 83% → 51% → 3%).

Three open questions that the current data does not answer:
1. **Predictive vs causal.** The LASSO finding shows PCs 2..k *correlate* with harm.
   It does not show they *cause* harm.
2. **Geometry of harm.** Is the harm signal AA-aligned, or genuinely orthogonal?
   Plan B used role-PCA as the basis without testing whether harm lives there.
3. **Defence vs blind spot.** Multi-axis defence (H2) was scoped out of Plan B.
   The blind-spot signal exists; whether capping the offending directions
   closes it is untested.

This document plans the next research moves. Ordering is conditional: cheap
diagnostics first, expensive runs only after they're justified by the
diagnostic outcome.

---

## Ext A — DiffMean `v_harm` diagnostic  *(do FIRST)*

**Goal:** identify what the LASSO is actually detecting. Compute the harm-label
contrast direction the same way the paper computes AA, but with harm/safe
labels instead of Assistant/role labels.

**Method.** From `results/plan_b_gemma2_27b/details.parquet`, take only the
`baseline` rows (n=500) at L*=21:
1. Split by binarised harm label: `harm_set` (74) and `safe_set` (426).
2. Compute `v_harm = mean(harm_set) − mean(safe_set)`, L2-normalize.
3. Report three numbers:
   - `cos_sim(v_harm, AA)` at L*=21
   - `||v_harm − P_K(v_harm)||² / ||v_harm||²` for K ∈ {3, 5, 10} role PCs
   - Single-direction AUC using `<h, v_harm>` alone vs LASSO multi-direction

**Output.** `src/analysis/harm_direction.py` (new) → JSON to
`results/plan_b_gemma2_27b/extensions/harm_direction.json` plus a small
markdown summary suitable for pasting into the project page.

**Cost.** ~1 hour. No model runs; pure parquet + safetensors compute.

**Predicted outcomes & implications.**
- `cos_sim(v_harm, AA) > 0.9`: harm is essentially the AA direction. The
  LASSO's +0.240 lift is mostly higher-PC noise. The H1 blind-spot framing
  weakens substantially; pivot to harm-PCA as the proper basis.
- `cos_sim(v_harm, AA) ∈ [0.5, 0.85]`: meaningfully novel direction exists
  alongside AA. Justifies Ext B (causal test) and reframes the result as
  "AA + harm-residual" rather than "AA + arbitrary PCs".
- `||v_harm − P_K(v_harm)||²/||v_harm||² > 0.3`: harm has structure *outside*
  the role-PCA subspace. Role-PCA is the wrong basis; harm-PCA should be
  computed on the harm-vs-safe split directly.

---

## Ext B — Causal test of `v_harm`  *(do IF Ext A says v_harm is meaningfully orthogonal to AA)*

**Trigger.** Run only if Ext A returns `cos_sim(v_harm, AA) < 0.85` AND
single-direction `<h, v_harm>` AUC > 0.85. Otherwise the predictive direction
isn't novel enough to be worth a causal test.

**Goal.** Distinguish predictive correlation from causal mechanism. The
LASSO/v_harm finding shows correlation. To establish causation, intervene on
the direction and check whether the outcome shifts.

**Method.**
1. Steer along `v_harm` at λ ∈ {0.1, 0.25, 0.5} on **clean baseline** prompts
   (no cap, no DAN). Just the 500 unsteered DAN prompts but steered.
2. Judge with the same Qwen 3.6 27B pass.
3. Acceptance: at least one λ produces a harm-rate increase ≥ 5 pp over
   baseline (14.8%) with BCa 95% CI not crossing baseline.

**Cost.** ~1.5 hours HF + judge for 3 conditions × 500 prompts.

**Predicted outcomes.**
- Harm rate climbs with λ → `v_harm` is causal for harm production. The
  H1+H2 paper has a clean story: "we found a direction that predicts harm,
  verified it causes harm under steering, and show that capping it as part of
  multi-axis defence closes the residual signal."
- Harm rate flat → `v_harm` is correlate-only. The LASSO is detecting
  something that *co-occurs* with harm (style? topic? response length?) but
  isn't a defence-actionable direction. This is itself a real finding —
  reframes the page from "blind spot" to "harm has linear structure but no
  single causal direction beyond AA."

---

## Ext C — Multi-axis defence H2  *(do AFTER Ext A, regardless of Ext B outcome)*

**Goal.** Test whether capping AA + PC2 + PC3 jointly closes the blind-spot
signal Plan B detected.

**Method.** Add one new condition to Plan B's sweep:
- `multi_axis_aa_pc2_pc3_capped` — `multi_axis_cap` from
  `src/steering/steerer.py`, capping AA + PC2 + PC3 each at p25 of their
  respective role-positive distributions, on the same layer range and
  position-policy as the working AA-cap.

**Acceptance criteria (any of which is informative).**
1. **Defence success:** LASSO AUC delta on multi-axis-capped activations
   shrinks from +0.240 to <+0.05; behavioural-bypass refusal rate stays
   ≥75% under PC2/PC3 attacks. → H2 confirmed in miniature.
2. **Higher-PC residual:** AUC delta unchanged or slightly reduced. Means the
   blind-spot signal lives in PC4-10 (which the LASSO selected); 3-axis
   defence is insufficient. Iterate to AA + top-K-LASSO-selected.
3. **Coherence collapse:** multi-axis cap breaks coherence the way the
   original 8-layer cap did. → 3 simultaneous caps at p25 each is too
   aggressive; calibrate per-axis tau more conservatively.

**Cost.** ~3-4 hours HF for one new condition + judge + analysis.

**Composition with Ext B.**
- If Ext B said `v_harm` is causal: also run `v_harm_cap` (cap along the
  learned direction directly) as a fourth defence variant. Compare to
  AA + PC2 + PC3. The data-weighted single-direction defence vs the
  hand-picked multi-axis defence is the cleanest engineering question.
- If Ext B said `v_harm` is correlate-only: skip `v_harm_cap`; multi-axis on
  PC2 + PC3 is the only defence variant worth running.

---

## Ext D — Behavioural bypass interpretation

**Goal.** Characterise *what* PC2/PC3 attacks actually do to model output
when they preserve coherence. Plan B's data shows refusal-style collapses
under PC attacks at coherence-preserving λ; we don't know what replaces it.

**Method.**
1. Filter PC2-attack and PC3-attack rollouts to coherent + non-refusal +
   non-harmful (`harm_label_primary ∈ {related_but_no_info,
   almost_enough_info, out_of_context}`).
2. For 20 random sampled responses per condition, hand-read and tag.
3. Compute embedding similarity of these responses to (a) baseline refusals,
   (b) baseline complies, (c) generic off-topic — using
   `sentence-transformers/all-MiniLM-L6-v2` or similar.
4. Tabulate: are PC-attack outputs evasive deflections, hedged partial info,
   meta-commentary about the prompt, or genuine off-topic?

**Cost.** ~2 hours total.

**Why it matters.** This is the most novel finding in Plan B's data and the
most underexplored. If PC3 has a clean semantic interpretation
("commitment-to-refusal-as-stance" / "voice-of-refusal"), it connects
directly to Anthropic's persona-vectors thread (Chen et al.) and provides
mechanistic interpretation alongside the predictive finding. Black-box
"orthogonal direction predicts harm" → semantic "PC3 governs refusal voice".

---

## Ext E — Coherence-matched random baselines

**Goal.** Make the "PC attacks vs random attacks" contrast clean. Plan B's
random baselines at λ=0.25 collapse the model to 100% nonsense, which is not
a fair comparison to PC attacks at the same λ (which preserve 33-56%
coherence).

**Method.** Sweep random-direction λ in {0.05, 0.10, 0.15} on 100-prompt
mini-runs to find the λ where randoms preserve ~50% coherence (matching PC2's
33% and PC3's 56%). Then run a full 500-prompt random condition at that λ.

**Cost.** ~2 hours HF (4 mini-runs × ~20 min + 1 full run × ~25 min) + judge.

**Why it matters.** Pre-deadline, the random-baseline claim on the page is
weak — the comparison is degenerate. Post-deadline, this fix lets you say
"at matched coherence, random directions do not collapse refusal style; PC2
and PC3 do." That's a stronger and more honest version of the H1 narrative.

---

## Execution order

```
                                        Ext A (DiffMean diagnostic)
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               │                             │                             │
   cos_sim>0.9 (v_harm ≈ AA)        cos_sim ∈ [0.5, 0.85]         residual >0.3 outside K
   AND single-dir AUC ≈ AUC_AA       (orthogonal harm signal)      (harm outside role-PCA)
               │                             │                             │
               ▼                             ▼                             ▼
   Skip Ext B.                        Run Ext B (causal test).      Skip Ext B.
   Reframe project: AA+higher-PCs    If harm rate ↑ with λ:        Pivot Ext C basis: cap
   is mostly noise; harm is 1D.      v_harm is causal.              along harm-PCA top-K
                                     Else: predictive only.         instead of role-PCA.
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
                                        Ext C (multi-axis defence)
                                             │
                                             ▼
                                Ext D + Ext E in parallel
                                (analysis-only / small runs)
```

---

## Total budget estimate

| Stage | Wall-clock | Compute | Notes |
| --- | --- | --- | --- |
| Ext A | ~1 hr | none | parquet + safetensors only |
| Ext B (if triggered) | ~1.5 hr | HF + judge | 3 conditions × 500 prompts |
| Ext C | ~3-4 hr | HF + judge | 1 new condition + analysis |
| Ext D | ~2 hr | embedding + manual | Plan B data + sentence-transformers |
| Ext E | ~2 hr | HF + judge | mini-sweep + 1 full condition |
| **Total** | **~8-10 hr** if Ext B fires; **~6.5-8.5 hr** if not | | |

A weekend's work, post-deadline.

---

## What we are NOT doing in this batch

- Adding subjects 2-4 (Qwen 3 32B, Gemma 4 31B ON/OFF) — that's the original
  Stage 2 post-deadline sweep (`plans/stage-2-infrastructure.md`).
- Shah-reconstructed dataset cross-check — same; original Stage 2 plan.
- Ordinal LASSO (T2.7b) — original Stage 2 plan.
- SGLang `--forward-hooks` spike (`plans/sglang_post_plan_b_spike.md`) —
  separate engineering thread.
- Capability eval (IFEval / MMLU-Pro / GSM8k / EQ-Bench) — original Stage 2
  T2.5; should run alongside Ext C at minimum to give multi-axis defence a
  Pareto axis.

These remain the scope of the broader post-deadline programme. This document
covers only the *theoretical* extensions surfaced during Plan B's review
discussion (predictive vs causal, harm geometry, multi-axis defence H2,
behavioural bypass).
