# May 3 Directive — Multi-Subject Execution

**Window:** 2026-04-29 → 2026-05-03 (~4.5 days)
**Goal:** Complete research across 4 core subjects for the May 3 fellowship application.

This file holds **only the deviations and sequencing decisions** that are not already in the stage plans. For task-level detail, follow the stage plan referenced in each phase below.

---

## Subject set (locked)

4 core subjects, bf16 / TP=4 on 4× RTX 5090:
1. Gemma 2 27B (Plan B locked — re-use cached results where possible)
2. Qwen 3 32B (thinking OFF)
3. Gemma 4 31B (thinking ON)
4. Gemma 4 31B (thinking OFF)

Excluded for this window (already deferred to Stage 7):
- Llama 3.3 70B → Ext 9 (hardware-gated; bf16 doesn't fit, fp8 path)
- Qwen 3.6-35B-A3B MoE → Ext 1 (tooling-gated; needs nnsight per-expert hooks)

---

## Cuts (the only deviations from stage plans)

These three cuts are the **safest** in the methodology-debt sense — each is pure post-hoc on cached data and can be added back later with zero rework.

| Cut | Rationale | Add-back path |
|-----|-----------|---------------|
| **Ext G capping-range sensitivity** | Paper Appendix F already validates this on Gemma 2 27B; marginal value low. | Run sensitivity sweep over `cap_layers` from cached activations. |
| **Ordinal LASSO (T2.7b)** | Binary LASSO is the headline metric (matches paper). Ordinal is robustness only. | Fit ordinal LASSO on existing `details.parquet`. No new rollouts. |
| **Gemma 4 cross-check judge across new subjects** | GPT-5.5 vs Qwen primary at 93% on Plan B is the validation we need. Re-running per-subject is generalization, not validation. Also: skip-self-judge rule means it would only apply to Gemma 2 27B + Qwen 3 32B (2 of 4) anyway. | Re-judge cached rollouts with Gemma 4 31B-it. No new generation. |

**Everything else in the stage plans + `extensions_post_plan_b.md` is kept**, including Ext A/B/C/D/E/F, Stage 5 across all 4 subjects, and capability eval × 4 subjects × 3 conditions.

---

## Pre-execution gates (Day 0)

These run before the multi-subject sweep starts, in this order:

1. **SGLang `--forward-hooks` spike** — see `plans/sglang_post_plan_b_spike.md`. 4-6 hr cost; if green, replaces HF-steered backend for the multi-subject Stage 4/6 runs (~12-15 hr saved at this scale). If red, fall back to HF-steered as in Plan B.
2. **`--subject` parameterization** — `plan_b.py` is Gemma-2-27B-hardcoded. Either refactor it to take `--subject` from `configs/subjects.yaml`, or split into a thin orchestrator over `configs/subjects.yaml`. Result-dir convention: `results/exp{N}_{name}/{subject_id}/...`.

Gate: do not start the multi-subject sweep until both are done.

---

## Phase mapping (refer to stage plans for task lists)

Each phase below maps to existing stage plans. The only new directive is the **subject loop** + the three cuts above.

### Phase A — Foundation (Exp 1 + Exp 2)
Refer to `plans/stage-3-foundation.md`. Run for each of the 4 subjects:
- Activation extraction at sweep layers, AA computation, PCA, layer-argmax via `cos_sim(PC1, AA)`
- Per-subject PC1≈AA validation (drop PC1-secondary analysis if `cos_sim < 0.7`)
- Per-PC harm correlation, blind-spot AUC delta (binary LASSO; ordinal cut per above)

Cached for Gemma 2 27B from Plan B — only re-run the 3 new subjects.

### Phase B — Attack (Exp 3 + Exp 4) + Ext E/F
Refer to `plans/stage-4-attack.md` + `plans/extensions_post_plan_b.md` (Ext E, Ext F).
- AA-cap baseline, orthogonal steering, adversarial null-space construction — per subject.
- **Ext E + F (coherence-matched random baselines + PC-attack λ sweep)** run once on Gemma 2 27B as the methodology fix; results referenced (not re-derived) for the other 3 subjects unless time permits.

### Phase C — Composition (Exp 5)
Refer to `plans/stage-5-composition.md`. Run for each of the 4 subjects.

### Phase D — Defence (Exp 6) + Ext A/B/C/D
Refer to `plans/stage-6-defense.md` + `plans/extensions_post_plan_b.md` (Ext A, B, C, D).
- Multi-axis cap (Phase A in stage plan) per subject.
- **Phase B in stage-6 (additive vs ortho-projected) stays conditional** on Phase A showing ≥10% additive AUC reduction (already conditional in stage plan — no new decision needed).
- **Ext A diffmean v_harm** per subject. **Ext B causal test** per subject (trigger: `|cos_sim(v_harm, AA)| < 0.85` AND `AUC(v_harm) > 0.85` — already fired on Gemma 2 27B; re-evaluate per subject).
- **Ext C multi-axis defence** is the same as Stage 6 multi-axis cap; do not double-execute. Treat Ext C as the "post-Plan-B framing" of Stage 6 Phase A.
- **Ext D behavioural bypass interp** runs once on the Gemma 2 27B Plan B rollouts; re-evaluate post-Phase D whether to extend to other subjects.

### Phase E — Capability eval
Refer to `configs/eval_sizes.yaml` + `src/evaluation/`. Run **4 subjects × 3 conditions** (unsteered + AA-cap + multi-axis-cap) on IFEval / MMLU-Pro / GSM8k / EQ-Bench. This is the denominator for the defence claim — do not skip.

### Phase F — Report + figures
Refer to `report/figures.md` + `src/visualization/figures.py::FIGURE_REGISTRY`. Pull metrics from each subject's `metrics.json`; aggregate into cross-subject panels.

---

## Sequencing caveats not in stage plans

- **Phase A and Phase C have no GPU dependency between subjects** — schedule subject N+1 extraction while subject N's PCA/analysis runs on CPU.
- **Phase B and Phase D share the steered-generation backend** (HF or SGLang post-spike). Co-schedule subjects through the same backend session to avoid load/unload churn.
- **Phase E (capability eval) is the longest single Phase by wall-clock** at multi-subject scale. Start it as soon as Phase D completes for any subject — does not need to wait for all 4.
- **Activation cache reuse:** Phase A's cached activations under `data/cache/activations/{model_id}/...` are consumed by Phases B/C/D. Do not delete between phases.
- **Gemma 4 31B thinking-ON vs thinking-OFF:** treat as two distinct subjects (separate `subject_id`s, separate result dirs), but extract activations at **both thinking tokens AND answer tokens** for the thinking-ON subject — compare PCAs (per CLAUDE.md "Models" → Tier 2 core).

---

## Stop conditions / fallback

If by **end of Day 3 (2026-05-02 morning)** Phase D is not complete for at least 2 subjects:
1. Skip Phase C (Stage 5 composition) on the remaining 2 subjects — keep it on the first 2 only.
2. Skip Ext B causal extension to subjects beyond Gemma 2 27B; cite Gemma 2 27B finding only.
3. Hold Phase E to whatever subjects have completed Phase D.

Capability eval on the completed subjects is non-negotiable.

---

## References

- `plans/plan.md` — stage overview & handoff protocol
- `plans/stage-2-infrastructure.md` — infra (Plan B mode locked; multi-subject extension via `--subject`)
- `plans/stage-{3,4,5,6}-*.md` — phase task lists
- `plans/stage-7-extensions.md` — Ext 1 (MoE), Ext 9 (Llama) deferral targets
- `plans/extensions_post_plan_b.md` — Ext A–G definitions, triggers, costs
- `plans/sglang_post_plan_b_spike.md` — SGLang `--forward-hooks` spike
- `plans/decisions.md` — unplanned-decision audit log (append cuts + spike outcome here)
- `plans/progress.md` — stage handoff blocks
- `CLAUDE.md` — locked decisions (model set, precision, judge, statistical framework)
