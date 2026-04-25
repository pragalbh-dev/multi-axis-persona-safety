# Plan B Directive — Stage 2 cutover for Anthropic fellowship deadline

**For the Stage 2 implementation agent.** This directive amends the existing Stage 2 plan to produce a **single-subject end-to-end results pass on Gemma 2 27B** within ~8 hours of GPU time, using only Stage 2 infrastructure. The output is a fellowship-application artifact with novel findings beyond the Lu et al. paper.

**Read first:** `CLAUDE.md`, `plans/CONVENTIONS.md`, `plans/stage-2-infrastructure.md`, `configs/inference_runtime.yaml`, `plans/decisions.md` (especially the 2026-04-25 fp8→bf16 entry), `plans/progress.md` (Stage 0→1 Handoff). All grid-tuned vLLM configs are in `configs/inference_runtime.yaml` — use them as-is.

---

## 1. Why this exists

**Anthropic fellowship deadline: 2026-04-26 evening.** ~30 hours from now. We don't have time to run the full Stage 3+4 sweep across all 4 subjects — that's ~140 GPU-hours / 5.8 days, scoped for the May 3 Astra & Constellation deadlines.

Plan B is the **smallest possible end-to-end run that demonstrates the H1 thesis with empirical signal on one subject** — Gemma 2 27B. Same code as the rest of the project, scoped tighter, runs in ~8 hours.

After the April 26 submission, the implementation continues normally and the full multi-subject sweep happens April 27 → May 3.

---

## 2. What to change in `plans/stage-2-infrastructure.md`

**Single-task amend: T2.9 only.** Everything else (T2.0 through T2.8, plus T2.1.5 / T2.1.6) stays exactly as currently planned and continues to be implemented as-is.

### Replace the existing T2.9 spec with this:

```markdown
- [ ] T2.9: Plan B run — single-subject H1 demonstration on Gemma 2 27B
  - **Why this is not a smoke test.** Plan B replaces the original ≤2-hour 100-prompt smoke. It runs at experiment-grade volumes (full DAN + Shah-reconstructed jailbreak sets, real PC2/PC3 steering) on Gemma 2 27B *only*, producing fellowship-deadline-ready results. Same code paths the original smoke would have exercised; just bigger volumes + more conditions. After the deadline, the full multi-subject sweep replays this exact pipeline across all 4 subjects (Stage 3 + Stage 4 proper).
  - **Subject:** Gemma 2 27B only (cheapest at bf16/TP=4 — 762 tok/s short, 542 tok/s long per `configs/inference_runtime.yaml`).
  - **Pipeline (sequential phases via `model_runner.run_in_subprocess`):**
    1. **Extraction (mini-T3.1).** 280 roles (275 + 5 default-Assistant variants) × **100 rollouts/role** (not paper's 300 — cheap τ-calibration is fine for one-subject preview). Output: per-rollout mean-response-token activations at every layer + per-rollout AA projections at the chosen capping layer. Use `short` profile from `configs/inference_runtime.yaml.gemma_2_27b.profiles.short`. Cache via `src/extraction/cache.py` to safetensors.
    2. **PCA + AA fit (mini-T3.3 / T3.1.5).** Run centered PCA on the 280 role-vector cache; pick the extraction layer L\* via argmax cos_sim(PC1, AA) per CONVENTIONS "Layer-scope convention"; keep top 10 PCs. Compute AA from default-Assistant rollouts vs role rollouts (paper §3.1 contrast-vector method). Save to `results/plan_b_gemma2_27b/extraction/{aa.safetensors, pcs.safetensors, eigenspectrum.npy}` with manifest noting L\* and cos_sim(PC1, AA).
    3. **Safety baseline (T3.6).** Full 1,100 prompts × 2 datasets (DAN + Shah-reconstructed) with no steering, no capping. Use `long` profile config from inference_runtime.yaml. Judge via Qwen 3.6-27B with `configs/judge_prompt.yaml` (T2.0 must finish first). Output: per-prompt harm label + binarized harm in `details.parquet`.
    4. **AA capping (T4.0 paper-verbatim).** Read Gemma 2 27B layer range from `configs/paper_capping_ranges.yaml` (T2.0 must transcribe from paper Appendix F before this step). If Appendix F doesn't ship Gemma 2 numbers, default to **center=middle-layer + 6, width=8, τ=25th percentile of the per-rollout AA-projection distribution** computed in step 1. Log this fallback to `decisions.md`. Run the same 1,100 × 2 jailbreak eval with AA capping active.
    5. **PC2 steering on AA-capped model.** λ ∈ {-2, 0, +2} (3 conditions). Same eval volume as step 4. Use steerer at the extraction layer L\* (single-layer, paper convention).
    6. **PC3 steering on AA-capped model.** λ ∈ {-2, 0, +2}. Same volume.
    7. **Random-direction control.** 5 random unit vectors, norm-matched to the lmsys-chat-1m mean residual-stream norm at L\* (cache it in step 1 as a byproduct), at λ=+2 only. Same eval volume.
    8. **Judge phase.** Single Qwen 3.6-27B pass over all responses from steps 3-7 using `judge` profile from inference_runtime.yaml (gpu_mem_util=0.75, max_seqs=512, prefix_caching=True, enable_thinking=False).
  - **Skip:** capability eval (T3.5.5), cross-check judge (T2.4 cross-check), LASSO-selected PCs (T3.8) — all deferred to the post-deadline full sweep.
  - **Output layout:** `results/plan_b_gemma2_27b/` with:
    - `config.yaml` — captured run config
    - `manifest.json` — schema, seed, git SHA, timestamps, artifact list
    - `details.parquet` — one row per (prompt_id, dataset, condition) with prompt, response, harm_label, harm_binary, projections onto PC1..PC10 + AA, condition tag
    - `metrics.json` — harm rate per condition (point estimate + BCa bootstrap 95% CI), Cohen's d for projection differences, paper-reproduction headline number
    - `extraction/` — AA + PC1..10 directions, eigenspectrum, lmsys norms
    - `figures/` — `harm_rate_per_condition.{png,html}`, `scree_plot.{png,html}`, `blind_spot_summary.{png,html}` (the H1 money plot)
  - **Acceptance criteria** (ALL must hold):
    1. AA capping reduces harm rate by **≥30 percentage points** vs unsteered baseline on the DAN dataset (paper-grade reproduction signal; paper reports ~60pp on Tier 1).
    2. At least one of {PC2 at λ=+2, PC3 at λ=+2} recovers **≥10 percentage points** of that reduction (i.e., harm rate climbs back ≥10 pp from the AA-capped baseline) with **bootstrap 95% CI excluding zero**.
    3. None of the 5 random-direction baselines at λ=+2 show ≥5 pp recovery (rules out "any nonzero steer breaks capping").
    4. cos_sim(PC1, AA) > 0.7 at L\* (paper threshold). If <0.7, log to decisions.md and report PC2/PC3 results against AA-only (drop PC1-based secondary analysis).
  - **Wall-clock budget: ≤8 hours on 4× RTX 5090 bf16/TP=4.** If gating, drop in this priority order: (a) PC3 conditions, (b) one of the random baselines down to 3, (c) trim extraction rollouts from 100→50/role.
  - **Implementation note:** every model load through `src/utils/model_runner.py::run_in_subprocess` (T2.1.6). The parent driver script (`src/experiments/plan_b.py` or similar) only orchestrates — never imports torch/vllm itself. This is the same vLLM-TP-teardown-leak finding from Stage 0; non-negotiable.
```

That's the only diff to `stage-2-infrastructure.md`. Don't touch the task dependency graph, the D1-D11 resolved decisions, or any other task.

---

## 3. What NOT to touch

- **T2.0–T2.8 specs and module APIs** — already correct. No edits to `src/{extraction,analysis,steering,evaluation,visualization,utils}/*.py` schemas beyond what the original Stage 2 plan dictates.
- **`configs/inference_runtime.yaml`** — locked from the grid search; consume the configs verbatim.
- **`configs/subjects.yaml`, `configs/eval_sizes.yaml`, `configs/paper_capping_ranges.yaml` (except Gemma 2 27B fill-in via T2.0), `configs/model_hooks.yaml`, `configs/role_expression_prompt.yaml`** — locked.
- **The Stage 3/4/5/6 plans** — no edits. Plan B doesn't merge them in; it just previews their results on one subject using Stage 2 infrastructure.
- **The Jailbreak Datasets policy in CONVENTIONS** — Plan B runs both DAN and Shah-reconstructed (subset), per the dual-dataset rule.

---

## 4. Implementation cutoff and ordering

**Hard deadline for Plan B kickoff: ~midnight tonight (2026-04-25 / 26 transition).** Plan B GPU run = ~8 hours; writeup + plots = ~3-4 hours; Anthropic deadline = April 26 evening. Math: kickoff midnight → done ~8 AM April 26 → writeup all morning → submit by evening. There's slack but not a lot.

### Order of finishing the underlying tasks (gating Plan B)

These must be **done before midnight** to allow Plan B to kick off:

1. **T2.0** (judge prompts YAML + Gemma 2 27B capping range transcription from paper Appendix F)
2. **T2.1.6** (model_runner subprocess wrapper — non-negotiable; see Stage 0 leak finding)
3. **T2.1** (extraction backend; HF forward-hooks per Stage 2 D2)
4. **T2.2** (PCA + AA fit + projections; Marchenko-Pastur threshold)
5. **T2.3** (steerer + capper at single-layer / multi-layer-range respectively)
6. **T2.4 main** (safety_eval orchestrator that consumes inference_runtime.yaml)
7. **T2.6** (results logger; details.parquet + manifest.json + metrics.json schema)
8. **T2.7 — partial** (BCa bootstrap + Cohen's d only; LASSO can wait)
9. **T2.8 — partial** (3 figures listed in Plan B; full plotting suite can wait)
10. **T2.9 = Plan B run itself** (replaces the old smoke test)

### Can wait until after April 26

- **T2.5** (capability eval) — Plan B skips capability eval. Implement post-deadline.
- **T2.7 LASSO** — Plan B uses raw PC2/PC3, not LASSO-selected. Implement post-deadline.
- **T2.4.5** (GPT-5.5 judge validation) — Plan B uses Qwen 3.6 judge directly. Validation is for the full sweep.
- **T2.4 cross-check judge phase** — Plan B uses primary judge only.

---

## 5. Plan B is pure Stage 2 — not pulling later stages

The H1 thesis is demonstrated by **invoking** existing Stage 2 modules with a particular condition set. No code from Stages 3-7 needs to be written. The condition set IS:

- Stage 3 T3.1 (extraction) → done at reduced rollout count, single subject
- Stage 3 T3.6 (safety baseline) → done on 1 subject
- Stage 4 T4.0 (AA capping) → done at paper verbatim, 1 subject
- Stage 4 T4.1 (per-PC steering) → done for PC2 + PC3 only, 1 subject, 3 strengths
- Stage 4 T4.2 (random vector baselines) → done at 1 strength, 1 subject

After April 26, the same calls run with full subject coverage during the post-deadline week. No code path from Plan B is throwaway.

---

## 6. The presentable findings (for the agent to expose in `metrics.json` and figures)

The H1 story the user will write up:

> *"On Gemma 2 27B, the Lu et al. AA capping defense reduces persona-jailbreak harm rate from {baseline}% to {capped}% on the DAN in-the-wild jailbreak set (paper reproduction). However, applying PC{2,3} steering at λ=+2 on top of the AA-capped model recovers {recovery_pp} pp of that reduction — demonstrating AA capping has a measurable blind spot along directions orthogonal to AA. Random-direction baselines at the same norm produce no such recovery, ruling out an unspecific steering effect. This is the first empirical demonstration that PC1-only/AA-only persona-space defenses leave attack surface; the full multi-subject characterization and adversarial null-space construction are forthcoming."*

`metrics.json` should expose every number the user needs to fill those blanks, plus 95% bootstrap CIs.

The 3 figures:
1. **`harm_rate_per_condition.{png,html}`** — bar chart, x = condition (baseline / AA-capped / +PC2 ±2 / +PC3 ±2 / +random ×5), y = harm rate %, error bars = BCa 95% CI. Stratify by dataset (DAN vs Shah-reconstructed) as paired bars or two panels. **This is the money plot.**
2. **`scree_plot.{png,html}`** — eigenspectrum of role-vector PCA, x = PC index, y = explained variance fraction, with Marchenko-Pastur threshold line. Conveys "persona space is high-dim."
3. **`blind_spot_summary.{png,html}`** — single-row dot plot or text card: "AA capping Δharm = X pp; +PC2 recovery = Y pp; random baseline recovery = Z pp." Useful as a slide.

---

## 7. Critical reminders

- **Subprocess every load.** vLLM TP=4 in-process teardown leaks ~25 GB/GPU between cells (Stage 0 finding). Use `src/utils/model_runner.py::run_in_subprocess` for every phase. Verify GPUs return to baseline (≤200 MiB used) between cells.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** on every child env (mitigates fragmentation; included in the grid-search wrapper, copy that pattern).
- **Pin `CUDA_VISIBLE_DEVICES=0,1,2,3`** via `from src.utils import env` at top of every script.
- **Seed everything** via `env.seed_everything(42)` — required for reproducibility per CONVENTIONS.
- **Never amend or force-push.** Per-task commits with `[Stage 2 / Tk.m] <brief>` prefix until Plan B; commit Plan B as `[Stage 2 / T2.9-plan-b] gemma 2 27B end-to-end H1 results`.
- **Log unplanned decisions** to `plans/decisions.md` using the existing template (e.g., if you fall back to a heuristic capping range for Gemma 2 27B, log it).

---

## 8. Post-deadline (informational only — do not start now)

After April 26 submission, the same Plan B pipeline replays on Qwen 3 32B, Gemma 4 31B (thinking ON + OFF) over April 27 → May 2, completing the multi-subject H1 picture for Astra & Constellation (deadline May 3). Stage 4 T4.0 Tier 2 capping sweep, T4.1 per-PC at full λ grid, T4.6 adversarial null-space, Stage 5 composition, Stage 6 multi-axis defense all follow. Total post-deadline budget: ~140 GPU-hours; available: ~144 hours.

The agent should **not implement these post-deadline tasks during Plan B prep**. Single focus: ship Plan B by April 26 evening with H1 signal on Gemma 2 27B.
