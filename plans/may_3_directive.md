# May 3 Directive — Multi-Subject Execution

**Window:** 2026-04-29 → 2026-05-03 (~4.5 days)
**Goal:** Complete research across 4 core subjects for the May 3 fellowship application.

This file holds **only the deviations and sequencing decisions** that are not already in the stage plans. For task-level detail, follow the stage plan referenced in each phase below.

> **Hardware amendment 2026-04-30 (see "Plan amendment" section at bottom).** Execution host is now **1× RTX PRO 6000 Blackwell 96 GB**, not the original 4× RTX 5090 / 128 GB. All subjects run **TP=1**, **never co-hosted** (always phased load → run → tear down), and the **cross-check judge is dropped entirely** (Gemma 4 31B-it cross-check is no longer in scope). The phase mapping below is unchanged in structure; the volumes / wall-clock estimates referenced in the stage plans need to be re-read with TP=1 in mind.

---

## Subject set (locked)

4 core subjects, bf16 / **TP=1 on 1× RTX PRO 6000 Blackwell 96 GB** (was: TP=4 on 4× RTX 5090; see amendment at bottom):
1. Gemma 2 27B (Plan B locked — re-use cached results where possible)
2. Qwen 3 32B (thinking OFF)
3. Gemma 4 31B (thinking ON)
4. Gemma 4 31B (thinking OFF)

**Co-hosting policy: never.** Even if a subject leaves headroom, the second model does not load alongside it. Every phase = load subject → run → tear down → load next. Same rule for the judge phase. This supersedes the CLAUDE.md "Inference & Serving" exception clause for the duration of this window.

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
| **Gemma 4 cross-check judge — DROPPED ENTIRELY** | GPT-5.5 vs Qwen primary at 93% on Plan B is the validation we need; re-running per-subject is generalization, not validation. **Hardened by amendment 2026-04-30**: cross-judge is no longer in scope at all (was a cut-with-add-back-path; now removed). Co-host ban + TP=1 reload cost makes the per-subject re-judge net-negative even as a follow-up. | Not re-runnable in this window. If demanded later, revive from cached rollouts on a future sweep. |

**Everything else in the stage plans + `extensions_post_plan_b.md` is kept**, including Ext A/B/C/D/E/F, Stage 5 across all 4 subjects, and capability eval × 4 subjects × 3 conditions.

---

## Pre-execution gates (Day 0)

These run before the multi-subject sweep starts, in this order:

1. **SGLang `--forward-hooks` spike** — see `plans/sglang_post_plan_b_spike.md`. 4-6 hr cost; if green, replaces HF-steered backend for the multi-subject Stage 4/6 runs (~12-15 hr saved at this scale). If red, fall back to HF-steered as in Plan B.
2. **`--subject` parameterization** — `plan_b.py` is Gemma-2-27B-hardcoded (defaults to `configs/plan_b.yaml`). Refactor to take `--subject {gemma_2_27b, qwen_3_32b, gemma_4_31b_thinking_on, gemma_4_31b_thinking_off}` resolving from `configs/subjects.yaml` + a per-subject overrides YAML. Result-dir convention: `results/exp{N}_{name}/{subject_id}/...`. Author the 3 new per-subject configs (volumes, expected L*, capping range; for thinking-ON also the thinking-vs-answer token-span extraction split). **Add a TP=1 load-smoke for each new subject** (≤30s; verifies bf16 + KV cache fit on the single 96 GB card before kicking off Phase A).

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
- **No cross-check judge.** Per Cuts table, Gemma 4 31B-it cross-judge is dropped entirely (not just per-subject). Headline harm numbers rest on Qwen 3.6-27B primary judge alone, validated against GPT-5.5 at 93% on Plan B.

### Phase E — Capability eval
Refer to `configs/eval_sizes.yaml` + `src/evaluation/`. Run **4 subjects × 3 conditions** (unsteered + AA-cap + multi-axis-cap) on IFEval / MMLU-Pro / GSM8k / EQ-Bench. This is the denominator for the defence claim — do not skip.

### Phase F — Report + figures
Refer to `report/figures.md` + `src/visualization/figures.py::FIGURE_REGISTRY`. Pull metrics from each subject's `metrics.json`; aggregate into cross-subject panels.

---

## Sequencing caveats not in stage plans

- **Phase A and Phase C have no GPU dependency between subjects** — schedule subject N+1 extraction while subject N's PCA/analysis runs on CPU. **TP=1 caveat:** because we cannot co-host (see policy above), the subject swap means a full vLLM/SGLang teardown + reload between subjects. Budget ~2-5 min per swap (model weights stay in pagecache, but engine init still runs).
- **Phase B and Phase D share the steered-generation backend** (HF or SGLang post-spike). Co-schedule subjects through the same backend session to avoid load/unload churn — i.e. complete all SGLang-viable subjects (Gemma 2, Qwen 3) on SGLang before swapping to HF for the Gemma 4 modes.
- **Phase E (capability eval) is the longest single Phase by wall-clock** at multi-subject scale. Start it as soon as Phase D completes for any subject — does not need to wait for all 4.
- **Activation cache reuse:** Phase A's cached activations under `data/cache/activations/{model_id}/...` are consumed by Phases B/C/D. Do not delete between phases.
- **Gemma 4 31B thinking-ON vs thinking-OFF:** treat as two distinct subjects (separate `subject_id`s, separate result dirs), but extract activations at **both thinking tokens AND answer tokens** for the thinking-ON subject — compare PCAs (per CLAUDE.md "Models" → Tier 2 core).
- **Judge phase is also TP=1 + standalone.** Qwen 3.6-27B judge loads after the subject tears down, classifies the parquet, tears down. No co-hosting.

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

---

## Plan amendment 2026-04-30 — single-GPU execution

**Trigger.** Execution host changed from the original 4× RTX 5090 / 128 GB total to **1× RTX PRO 6000 Blackwell 96 GB** (sm_120, driver 580.142, CUDA toolkit 12.9). Confirmed by `nvidia-smi` and the SGLang spike artifacts (`plans/sglang_post_plan_b_spike.md` Results section). The directive's compute assumptions need to be re-grounded.

**Amendments to the plan above (all binding for this window):**

1. **Tensor parallel = 1 across all subjects + judge.** `subjects.yaml` already auto-clamps `tensor_parallel_size` via `torch.cuda.device_count()` at load time, so the YAML field stays at 4 but the runtime value is 1. No edit to `subjects.yaml` required for that field. **Hardware fit must be verified per subject** before Phase A — Qwen 3 32B and Gemma 4 31B at bf16 + KV cache on a single 96 GB card is tight (weights ≈ 62 GB; KV cache budget ≈ 25-30 GB at `gpu_memory_utilization=0.90`). The Gate #2 smoke loads (see Pre-execution gates) are the canary. If a subject won't fit, log to `decisions.md` and pick a fallback (drop seq len, lower `gpu_memory_utilization`, or fp8 — fp8 is a CONVENTIONS deviation that needs an explicit decision entry).

2. **No co-hosting, ever.** Even when a subject leaves headroom, the next model does not co-load. Always: load → run → tear down → load next. CLAUDE.md "Inference & Serving" exception clause is suspended for this window. Same rule applies to subject↔judge swaps. Budget ~2-5 min per swap.

3. **Cross-check judge dropped entirely.** Gemma 4 31B-it cross-check is removed from scope, not just per-subject. Headline harm numbers rest on Qwen 3.6-27B primary alone (validated to 93% vs GPT-5.5 on Plan B, sufficient per paper's 91.6% bar). The Cuts table above is updated with the harder framing; the "add-back path" column is now N/A for this window.

4. **Wall-clock implications.** TP=1 vs original TP=4 is roughly 3-4× slower on compute-bound phases (extraction + judging). The ~4.5 day window for 4 subjects becomes infeasible at that scale. Revisit volumes per phase as the agent picks each up; expect cuts to (a) extraction rollouts (drop from 100/role → 30/role per Plan B precedent), (b) capability eval breadth (cut MMLU-Pro 1400 → seeded 500), or (c) Phase C composition coverage (1-2 subjects instead of 4) under the existing Stop-conditions / fallback section. Don't pre-commit; let each phase agent re-cost with measurements.

5. **SGLang spike verdict still holds** even though it ran on this same 1-GPU host (the spike's TP=1 numbers are the relevant baseline now, not the directive's TP=4 estimates). Gemma 2 + Qwen 3 stay on SGLang for steered phases; Gemma 4 modes stay on HF.

**Source.** Conversation 2026-04-30 between user (pragalbh-dev) + agent during Gate #2 kickoff. Logged in `plans/decisions.md` 2026-04-30 entry.

---

## Plan amendment 2026-05-03 — post-Phase-B retrim

**Trigger.** Phase A complete (all 4 subjects, see `progress.md` 2026-05-01 entry) and Phase B complete (attack arm + first multi-axis-cap shot for 4 subjects, see `results/phase_b/<subject>/headline.json`). The May 3 fellowship deadline is past; we are no longer time-boxed. Re-scoping the remaining work around what the Phase B headlines actually justify — drop work that doesn't move the narrative, keep work that addresses what the data already surfaced.

**Phase B headline summary that drives this re-scope:**

| Subject | baseline → AA-cap | Strongest attack recovery (Δ vs AA-cap) | cos(v_harm, AA) | What this implies |
|---|---|---|---|---|
| Qwen 3 32B | 8.6% → 2.4% | All ≤ −1.8 pp (no recovery) | 0.13 | AA cap is sufficient. Multi-axis adds nothing. |
| Gemma 4 31B think-OFF | 10.2% → 11.2% (**AA cap fails**) | adv-null +7.5 pp; PC3 +12.4 pp | 0.05 | The strongest H1+H2 finding in the project. |
| Gemma 4 31B think-ON | 15% → 5.1% | PC2 +1.4 pp | 0.56 | AA absorbs most of harm; small residual blind spot. |
| Gemma 2 27B (Plan B) | 14.8% → 1.8% | (Plan B baseline; replicated) | (per Plan B) | Cached. |

**Amendments (binding from this point forward):**

### 1. Cuts hardened

| Cut | Status | Rationale |
|-----|--------|-----------|
| **Phase C — Stage 5 Composition (4 subjects)** | **DEFERRED to extension** (was: per-subject in Phase C) | H3 (linearity vs manifold) is not what Phase B's results speak to. Persona arithmetic doesn't sharpen the g4_off blind-spot story or the thinking-mode geometry shift. Moves to `plans/extensions_post_plan_b.md` as Ext H — pick up later if a reviewer asks for H3 evidence or as natural follow-up. |
| **Stage 6 / Phase D scoped to g4_off only** | **NEW SCOPE** (was: 4 subjects × 4 multi-axis configs) | Qwen: AA alone is sufficient + multi-axis collapses coherence (28% nonsense at pos1). g4_on: blind spot is small (PC2 +1.4 pp). Only g4_off has a genuine multi-axis question to answer ("can AA+PC2+PC3 close the 12.4 pp PC3 attack recovery?"). Skip multi-axis calibration on the other three. |
| **Stage 6 Phase B (cross-percentile τ sweep)** | **CUT** (was: conditional on ≥10% Phase A gain) | Decision rule's premise (Phase A multi-axis additive gain) only fires for g4_off, and even there a per-direction τ + layer-range fit is sufficient. 5^k cross-percentile grid does not pay back. |
| **Phase E capability eval** | **TRIMMED** (was: 4 subjects × 3 conditions × 4 benches) | Run 4 subjects × 2 conditions {unsteered, AA-cap} on IFEval + GSM8k + EQ-Bench. Add `multi-axis-cap` as a third condition only on g4_off. Drop MMLU-Pro entirely (or seeded 500 if a reviewer pushes back — flag in writeup). |
| **Ext B causal v_harm test** | **SCOPED** to g4_off + g4_on | Cos(v_harm, AA) = 0.05 (g4_off) and 0.56 (g4_on) are the two subjects where v_harm geometry is interesting enough to justify a causal probe. Skip on Qwen (cos=0.13 but AA already covers harm) and on Gemma 2 27B (cached / different basis run). |
| **Ext D bypass interpretation** | **KEEP** — analysis-only on existing rollouts | No new GPU runs. Filter `phase_b/<subject>/lambda_sweep.parquet` for coherent + non-refusal + non-harmful PC2/PC3-attack rows; embed via sentence-transformers; classify what replaces refusal. Connects PC3 to a semantic identity ("voice-of-refusal") which the project currently lacks. |
| **Thinking-vs-answer activation split** | **DEFERRED** to `stage-7-extensions.md` Ext 2 | The `extract_thinking_answer_split: true` flag in `configs/plan_b_gemma_4_31b_thinking_on.yaml` is not honored by `src/extraction/backend_hf.py` — Phase A's PCA on g4_on was on pooled thinking+answer tokens. The L*=59 result therefore conflates the two spans. **The current Phase A/B numbers stand for now**; the split is a Stage-7 follow-up that will refit PCA per span and re-derive AA / blind-spot lift. Logged separately so it is not forgotten. |

### 2. What remains (new scoped plan)

The remaining work is now five threads, ordered by what most strengthens the narrative:

**A. Stage 6 / Phase D scoped to Gemma 4 31B thinking-OFF only.** Per-direction τ calibration for PC2 + PC3 on a 200-prompt validation subset (sweep τ ∈ {1, 10, 25, 50, 75}th percentile per axis at the existing capping layer range). Pick the (τ_PC2, τ_PC3) that minimizes harm under PC3 attack (currently 23.6%) without dropping coherence below 90%. Then run a full 550-prompt test-split evaluation of `{AA-cap, AA+PC2-cap, AA+PC2+PC3-cap}` against the unsteered + adv-null + PC3-attack baselines. Output: `results/phase_d/gemma_4_31b_thinking_off/{multi_axis_calibration.json, test_split.parquet}`. **This is the H1+H2 headline strengthener.**

**B. Ext B causal v_harm test on g4_off + g4_on.** Steer along `v_harm` at λ ∈ {0.1, 0.25, 0.5} on the 500 baseline DAN prompts (no cap, no DAN-style attack, just `v_harm` steered). If harm rate climbs with λ → v_harm is causal. Rules out the "v_harm is correlate-only" outcome for the two subjects where v_harm geometry is non-trivial. Outputs land at `results/phase_b/<subject>/extensions/v_harm_causal/`.

**C. Phase E capability eval (trimmed).** 4 subjects × {unsteered, AA-cap} × {IFEval, GSM8k, EQ-Bench}. Add `multi-axis-cap` (best config from thread A) on g4_off only. ~9 subject-condition cells. Standard `src/evaluation/capability.py` driver.

**D. Ext D bypass interpretation.** Pure analysis on `phase_b/<subject>/lambda_sweep.parquet` + the Plan B Gemma 2 27B rollouts. Filter, embed, hand-tag; tabulate refusal-replacement modes per axis.

**E. Phase F report figures.** Cross-subject aggregation. Some cross-model figures already exist (`docs/figures/cross_model_*`); re-run `scripts/generate_cross_model_figures.py` after thread A lands so the multi-axis-defence panel reflects the calibrated config.

### 3. Time estimates

All on 1× RTX PRO 6000 Blackwell 96 GB, TP=1, sequential. Subject swaps cost ~2-5 min each.

| Thread | Compute | Wall-clock | Notes |
|---|---|---|---|
| **A. Stage 6 / Phase D (g4_off only)** | HF + judge | **8-10 hr** | 200-prompt validation × 5 τ × 2 axes ≈ 2000 generations + 550-prompt test-split × 3 multi-axis configs ≈ 1650 generations. Plus τ-fit analysis (~30 min) + judge passes. |
| **B. Ext B causal v_harm (g4_off + g4_on)** | HF + judge | **5-7 hr** | g4_off: 3 λ × 500 prompts ≈ 1500 generations + judge ≈ 2 hr. g4_on: same volume but ~2× slower due to thinking traces ≈ 3-4 hr. |
| **C. Phase E capability (trimmed)** | vLLM + grading | **20-25 hr** | 9 cells × ~1700 prompts × per-bench cost (IFEval + GSM8k cheap; EQ-Bench moderate). g4_on cell ~2× slower; multi-axis on g4_off adds 3 more cells. Subject-load swap overhead included. |
| **D. Ext D bypass interp** | CPU only | **2-3 hr** | sentence-transformers embedding + manual tagging on existing parquets. No GPU. |
| **E. Phase F report figures** | CPU only | **3-5 hr** | Figure regen + writeup integration. Mostly authoring time. |
| **Total remaining** | | **~38-50 hr compute + ~5-8 hr writeup** | ≈ 2-3 days of wall-clock if threads are run sequentially, less if D + E are interleaved with A's GPU work. |

### 4. What was deferred (extension log)

These are out of scope for this re-trimmed plan but should not be forgotten:

- **Composition (Phase C / Stage 5)** → moves to `plans/extensions_post_plan_b.md` as a new entry "Ext H — Persona composition (H3)". 4 subjects × 50 pairs × 200 rollouts × per-pair PCA = ~12-15 hr if/when picked up.
- **Thinking-vs-answer activation split (Gemma 4 31B thinking-on)** → `plans/stage-7-extensions.md` Ext 2 (already exists in spirit). Requires patching `src/extraction/backend_hf.py` to honor `extract_thinking_answer_split` (tokenizer-driven span detection over the response), refitting PCA per span, recomputing AA / blind-spot lift on the answer-only subset. ~4-5 hr.
- **MMLU-Pro on Phase E** → revivable on demand.
- **Phase E multi-axis capability evaluation on Qwen + g4_on** → not informative given those subjects don't benefit from multi-axis defence, but flag if a reviewer asks for symmetric capability numbers.

### 5. Stop conditions / fallback (relaxed)

Original directive's Day-3 stop conditions are voided — no deadline pressure. The re-scoped threads above are independently runnable; if any one returns surprising numbers, we re-plan rather than truncate.

**Source.** Conversation 2026-05-03 between user (pragalbh-dev) + agent. Logged here + a one-line entry in `plans/decisions.md`.
