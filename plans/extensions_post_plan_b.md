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

## Known issues with the current run (to address post-deadline)

The Plan B re-run produced clean headline numbers, but six methodological
gaps and data-quality concerns surfaced in review. Listed here so they
don't get lost; each is mapped to which extension (if any) addresses it.

### 1. Coherence loss under PC and random attacks
- PC2 attack at λ=0.25: **67.2% nonsense** (32.8% coherent).
- PC3 attack at λ=0.25: **44.0% nonsense** (56.0% coherent).
- Random attacks at λ=0.25: **100% nonsense** (0% coherent).

**Implication.** The "0% harm under PC attacks" claim is partly attributable
to the steering perturbation degrading coherence — there is less coherent
text to be judged — not purely to the cap suppressing harmful information.
The "behavioural bypass" refusal-rate metric is also inflated by nonsense
mass on the full denominator (which is why the page reports both raw and
coherence-adjusted versions).

**Addressed by:** Ext E (coherence-matched random baselines) AND Ext F
(coherence-matched PC-attack sweep, new — see below). Together these give
a clean "at matched coherence X%, PC attacks reduce refusal-rate by Y pp;
random baselines do not" claim that the current data cannot support.

### 2. Within-cap Cohen's d has small harm-positive group
- AA-cap reduces harm to **n=9 of 500** (1.8% rate is the success of the
  defence; the side effect is that within-cap Cohen's d on PC2/PC3 separating
  the 9 harm cases from 491 non-harm uses an unbalanced sample).
- CI is reported (PC2: [-2.26, -1.11]; PC3: [-1.91, -0.96]) but the
  underlying n=9 limits how tight it can get, regardless of how many
  bootstrap resamples we run.

**Implication.** The Cohen's d numbers on the page are large but the CI
breadth invites a reviewer to discount them. Two options for tightening:
(a) accept the wide CI honestly and report — current page does this; (b)
pool capped runs across all four post-deadline subjects (Qwen 3 32B, Gemma 4
31B ON/OFF) to grow the harm-positive group. If Gemma 2 27B's 1.8% post-cap
rate is representative, four subjects × 500 prompts ≈ ~36 capped harm
cases, much tighter CI.

**Addressed by:** option (b) is implicit in Q5 / cross-subject replication.
Not a standalone extension.

### 3. Cross-judge validation — DONE at 93%
- GPT-5.5 cross-check ran on a 200-sample stratified subset of
  `rollouts_all_judged.parquet`. **Binary agreement vs Qwen 3.6-27B primary
  judge = 93%**, matching the paper's 91.6% deepseek-v3-vs-human bar.
- `metrics.json::judge_agreement_gpt55_vs_primary` field still shows
  `null` (write-back wasn't wired during the deadline rush) — should be
  patched to `0.93` for the page record. Underlying labels live in the
  GPT-5.5 sample parquet from the validation run.

**Implication.** Judge confidence is established. Every harm number on
the page rests on a judge that agrees with a strong frontier model 93%
of the time. The α-track's headline-defensibility question is now
purely about the EXPERIMENTAL setup (coherence, λ choice, basis), not
the judge.

**Status:** locked. Do NOT re-run — the budget is spent and the result
is sufficient. CLAUDE.md "Evaluation" section updated 2026-04-26 to
reflect this.

### 4. Capping layer range was a paper-fallback, not paper-validated
- Lu et al. publish capping configs for Qwen 3 32B (layers 46–53, 72–83%
  depth) and Llama 3.3 70B (56–71, 70–89% depth) but **not for Gemma 2
  27B**. We chose [33, 38] = 71.7–82.6% depth by proportional matching to
  Qwen.

**Implication.** This worked (cap defends, coherence preserved), but it's
methodologically a guess. The headline numbers could shift modestly under
better-tuned ranges (narrower vs wider, deeper vs shallower).

**Addressed by:** **Ext G — capping-range sensitivity sweep** (~1 hr HF).
Not high-priority unless Ext A says the harm geometry question dominates.

### 5. τ calibration is per-axis but the n=30 default-Assistant sample is small
- The current cap uses τ = p25 of the role-positive projection of n=30
  default-Assistant rollouts at the capping layers. With n=30 the percentile
  estimate is noisy.
- For Ext C (multi-axis defence), we'll need analogous τ calibration for
  PC2 and PC3 in their role-positive directions. Same noise concern, three
  times over.

**Implication.** Convention should be locked before Ext C: either widen the
calibration set (resample paper's full 275-vector cache through this
model's L*=21 forward), or use a more robust statistic (mean ± 1 SD, or
trimmed mean, instead of p25). Document the choice in CONVENTIONS before
running Ext C.

**Addressed by:** prerequisite documented in Ext C; not a standalone
extension.

### 6. PC6 is uninformative; PC1 is redundant — both interesting, neither characterised
- LASSO selects PC2, 3, 4, 5, 7, 8, 9, 10. Misses **PC1** (expected — it's
  redundant with AA, which is a separate feature; this is empirical
  confirmation of the paper's §3.1 claim) and **PC6** (unexpected — what
  makes PC6 different from its neighbours?).

**Implication.** PC6 not being selected is a small but real signal about
role-PCA structure that the project hasn't used yet. Could be: (a) PC6
captures a pure-style axis with no harm correlation, (b) PC6 is the noise
edge where the role-PCA basis becomes unstable, (c) there's a structural
reason we haven't identified.

**Addressed by:** falls naturally out of Ext D (behavioural-bypass
interpretation) if we extend it to project responses onto each PC
individually and tag what each PC governs. Worth a paragraph in the writeup
even if no extra experiment is run.

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

## Ext F — Coherence-matched PC-attack λ sweep

**Goal.** Resolve the coherence-attribution ambiguity in the headline. The
current run uses a single λ=0.25 calibrated on residual norms, but we don't
know the coherence-vs-bypass Pareto frontier for PC2 and PC3 attacks
individually. A reviewer's natural question — "is the refusal-rate drop
just the model degenerating, or a real behavioural shift?" — needs a clean
answer that the current data cannot give.

**Method.**
1. Sweep PC2 and PC3 attack λ ∈ {0.05, 0.10, 0.15, 0.20, 0.25} on a
   100-prompt mini-run (10 conditions × 100 prompts = 1,000 generations).
2. For each (axis, λ) cell, report: harm rate, coherence (1 − nonsense
   rate), refusal-keyword rate (raw), refusal-keyword rate (coherent only).
3. Pick the λ for each axis where coherence ≥ 80% (matching AA-cap's 91%
   loosely). Run a full 500-prompt condition at that λ.
4. Compare: at coherence ≥ 80%, does the cap's refusal pattern still
   collapse under PC2/PC3 steering, or does it survive?

**Cost.** ~3 hr HF (10 mini-conditions × ~12 min + 2 full conditions ×
~20 min) + judge.

**Predicted outcomes.**
- *Refusal still collapses at matched coherence* → behavioural bypass is a
  real finding, robust to coherence accounting. The page's main claim
  survives stricter scrutiny.
- *Refusal partially recovers when coherence is preserved* → some of the
  bypass on the current page is coherence-driven, not bypass-driven. The
  refined claim narrows: "PC attacks at λ ≥ X push the model into
  coherence-loss territory faster than into bypass territory." Still
  publishable but reframed.
- *Refusal fully recovers at matched coherence* → there is no behavioural
  bypass at coherence-preserving λ; the apparent bypass at λ=0.25 was
  pure coherence loss. This would be a null result and would weaken the
  page's H1 narrative substantially.

**Why it matters more than Ext E.** Ext E fixes the *random baseline* arm
of the comparison (cosmetic). Ext F fixes the *experimental arm* of the
comparison (substantive). Of the two, Ext F is the one that can change
whether the headline finding survives review. Run Ext F first; Ext E
becomes a follow-up only if Ext F's results are clean.

---

## Ext G — Capping-range sensitivity sweep

**Goal.** Confirm the chosen capping range [33, 38] (71.7–82.6% depth) is
locally optimal for Gemma 2 27B. The paper publishes capping configs for
Qwen 3 32B and Llama 3.3 70B; we picked [33, 38] by proportional matching
to Qwen. This worked, but the sensitivity to layer choice is unmeasured.

**Method.** Mini-run (100 prompts) of AA-cap at four alternative ranges:
- [34, 36] (narrower, centred)
- [31, 40] (wider, same centre)
- [37, 41] (deeper)
- [29, 34] (shallower)

For each, report harm rate, coherence rate, refusal-keyword rate. Compare
to the working [33, 38] baseline.

**Cost.** ~1 hr HF (4 conditions × ~12 min) + judge.

**Predicted outcomes.**
- *All four alternatives produce comparable defence + coherence numbers*
  → [33, 38] is in a stable plateau; the proportional-matching heuristic
  was sound. Headline numbers are robust.
- *One alternative produces meaningfully better numbers (lower harm + ≥90%
  coherence)* → the headline numbers are conservative; better-tuned cap
  exists. Re-run the full Plan B with the new range before any further
  extension. Worth doing.
- *All alternatives produce coherence collapse* → [33, 38] is the only
  working window; defence is fragile. Reframe the project around "narrow
  effective-cap window on Gemma 2 27B."

**Why it matters.** Lower priority than Ext F or Ext A. Run only if Ext A
confirms the role-PCA basis is the right one (otherwise basis question
dominates and layer-range tuning is moot).

---

## Execution order

Two parallel tracks: **(α) headline-defensibility** (does the current page's
claim survive scrutiny?) and **(β) project-direction** (what should the next
phase pursue?). Run α-track first — these can change whether the page's
claim holds. β-track follows.

```
α — headline-defensibility (run first; can refute/refine current findings)

   GPT-5.5 cross-judge on existing rollouts (Issue #3, ~30 min, $5-10)
                              │
                              ▼
                Ext F — coherence-matched PC-attack λ sweep
                       (Issue #1, ~3 hr; substantive)
                              │
                              ▼
                Ext E — coherence-matched random baselines
                       (Issue #1 cosmetic complement, ~2 hr)
                              │
                              ▼
                Ext G — capping-range sensitivity (Issue #4, ~1 hr;
                       only if Ext A says role-PCA basis is the right one)


β — project-direction (run after α-track confirms findings)

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
                                Ext D (behavioural-bypass interpretation,
                                       analysis-only)
```

---

## Total budget estimate

| Stage | Track | Wall-clock | Compute | Notes |
| --- | --- | --- | --- | --- |
| GPT-5.5 cross-judge | α | ~30 min | OpenAI API (~$5-10) | judging only, no new generation |
| Ext F (PC-attack λ sweep) | α | ~3 hr | HF + judge | 10 mini × 100 prompts + 2 full × 500 |
| Ext E (random-baseline λ sweep) | α | ~2 hr | HF + judge | 4 mini × 100 prompts + 1 full × 500 |
| Ext G (capping-range sensitivity) | α | ~1 hr | HF + judge | 4 mini × 100 prompts |
| Ext A (DiffMean diagnostic) | β | ~1 hr | none | parquet + safetensors only |
| Ext B (causal test, if triggered) | β | ~1.5 hr | HF + judge | 3 conditions × 500 prompts |
| Ext C (multi-axis defence) | β | ~3-4 hr | HF + judge | 1 new condition + analysis |
| Ext D (bypass interpretation) | β | ~2 hr | embedding + manual | existing data + sentence-transformers |
| **Total** | | **~14-16 hr** if everything fires; **~12-14 hr** if Ext B / Ext G skip | | |

Two weekends' work, post-deadline. α-track is ~6.5 hr alone and gates
whether any β-track effort is worth investing.

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

---

## Ext H — Persona composition (H3)  *(deferred 2026-05-03 from Phase C)*

**Status.** Was Phase C of `plans/may_3_directive.md`; moved here by the
2026-05-03 retrim amendment. Picked up only if a reviewer requests H3
(linearity vs manifold) evidence or as natural follow-up after the H1+H2
narrative ships.

**Goal.** Test whether persona vector arithmetic is linear or shows nonlinear
interactions (paper §3.1 + PERSONA paper 2602.15669 trait arithmetic, scaled
to multi-trait personas).

**Method.** Per `plans/stage-5-composition.md` T5.1–T5.7. Locked α=β=0.5
linearity test (R² + residual norm + per-PC residual projection); cap-bypass
under composition; ~30-50 persona pairs across complementary / contradictory
/ neutral / safety-concerning categories.

**Cost.** ~12-15 hr at TP=1 across 4 subjects (50 pairs × 200 rollouts ×
4 subjects ≈ 40K generations, plus PCA fits + analysis). Single subject
~3-4 hr if scoping.

**Why deferred.** Phase B's results (g4_off AA-cap failure + thinking-mode
geometry shift) define the project's H1+H2 narrative. H3 doesn't sharpen
either thread. If the writeup gets a reviewer ask for "what about non-linear
persona composition?", run this then — otherwise the existing report is
self-contained.

---

## Ext I — Thinking-vs-answer activation split  *(deferred 2026-05-03; cross-references stage-7-extensions Ext 2)*

**Status.** Was scoped in `configs/plan_b_gemma_4_31b_thinking_on.yaml` via
`extract_thinking_answer_split: true`, but `src/extraction/backend_hf.py`
never honored the flag. Phase A's PCA on `gemma_4_31b_thinking_on` was fit
on **pooled thinking + answer tokens**. The L*=59, cos_sim(PC1, AA)=0.822,
blind-spot-lift=0.101 numbers therefore conflate the two spans.

**Goal.** Refit PCA per span (thinking-only, answer-only) and re-derive AA
+ blind-spot lift on each. Check whether L*=59 is driven by the thinking
trace dragging the PC1≈AA direction toward the very-late layers (in which
case the answer-only L* should land closer to the ~50% depth band of
g4_off's L*=14) or whether thinking-on truly concentrates safety geometry
at the model's tail.

**Why this matters.** The L*=59 vs L*=14 gap on identical weights is the
most novel cross-subject finding in the project. As it stands, it's not
defensible against the natural critique "your PCA was fit on a mixture of
thinking traces (which dominate the token budget) and answer tokens — the
late L* might just be where thinking traces concentrate, not where harm
relevance lives." The split removes that ambiguity.

**Method.**
1. Patch `src/extraction/backend_hf.py` to honor `extract_thinking_answer_split`.
   Span detection runs over the response: thinking-tokens span = between the
   model's `<start_of_turn>model` ... `<end_thinking>` markers (Gemma 4
   chat template); answer span = after `<end_thinking>` until `<end_of_turn>`.
   Save per-rollout mean activation for each span separately.
2. Refit PCA on thinking-only and answer-only activation matrices. Each
   subset is the same 30 rollouts × N roles, so n shrinks per subset (from
   ~30 × 463 = 13.9k to ~30 × 463 each — same n, different signal).
3. Recompute AA per span, per layer; pick L* per span (argmax cos_sim).
4. Recompute LASSO blind-spot lift per span on the 500-prompt baseline.
5. Compare: does answer-only L* shift back to mid-depth? Does AA(answer-only)
   geometry look like g4_off's? Does the blind-spot lift on thinking-only
   resemble the all-tokens result?

**Cost.** ~4-5 hr — extraction backend patch + one re-run of step_1b on
g4_on with span tagging + downstream PCA / AA / blind-spot recompute.
Full plan_b.py orchestration; no judge re-runs needed (existing Phase A
rollouts are reusable, only activations change).

**Output.** `results/phase_a/gemma_4_31b_thinking_on/extraction/{thinking,answer}/`
with the standard Phase A artifact set (aa.safetensors, pcs.safetensors,
L_star.txt, per_layer_cos_sim.json, pca_meta.json).

**Why deferred.** Current Phase A/B headline numbers stand without the
split — the split refines the interpretation rather than refuting the
finding. Doing it now would require rolling back Phase B for g4_on (which
would consume cached AA / PCs that are about to change). Cleaner to ship
the H1+H2 story on the current artifacts and run the split as a follow-up
"are these results stable under per-span PCA?" sanity check.

**Cross-reference.** This is the concrete realisation of the "reasoning
subspace deep-dive (thinking-vs-answer geometry)" extension already
sketched in `plans/stage-7-extensions.md` Ext 2 (per progress.md
2026-04-24 17:00 entry). Whichever doc is picked up first should
cross-reference the other.
