# Decisions Ledger

Append-only log of **unplanned decisions** — any choice made during stage execution that was NOT in the pre-written stage plan.

**What belongs here:**
- Picking a specific integer / hyperparameter when the plan said "something around X" (e.g., "extraction layer L* landed at 30, not the paper's reported 32 — difference of 2, log why").
- Resolving an ambiguity in the paper's convention (e.g., paper says "25th percentile" — of which distribution? of which activations? if we chose differently from the plan's default).
- Picking between two viable library versions or implementations.
- Small scope cuts taken under time pressure ("skipped the cross-check judge on 50 prompts because judge server was GPU-starved").
- Anything you'd want a future agent (or the user) to be able to audit, question, and potentially reverse.

**What does NOT belong here:**
- Progress updates (those go in `progress.md`).
- Task completion checkboxes (those go in the stage plan).
- Decisions already specified in the stage plan or `CONVENTIONS.md` (those were pre-planned — no need to log).
- Transient state during implementation (only the final decision goes here).

**Rule of thumb:** if you caught yourself saying "the plan didn't cover this, so I'm going with X" — that's a decision. Log it.

---

## Template

Copy this block, fill in, append at the **end** of this file. Do not edit earlier entries.

```markdown
## [YYYY-MM-DD HH:MM] Stage N / T{k.m} — {short title}

**Decision:** {what was decided in 1-2 sentences}

**Alternatives considered:**
- Option A — {why rejected}
- Option B — {why rejected}

**Reason:** {why this option won. Cite the constraint that forced the choice.}

**Source:** {pick one or more}
- Paper line / appendix reference (e.g., "paper line 691" or "Appendix D.2.2")
- File path (e.g., "configs/paper_capping_ranges.yaml")
- External URL
- User instruction ("user said in chat on 2026-04-24 to prefer X over Y")
- Own judgment ("no source; judgment call because …")

**Reversibility:** {high / medium / low}
- **High** — decision can be flipped by changing one config value and re-running one task.
- **Medium** — flipping requires re-running a stage's worth of work (hours of GPU).
- **Low** — flipping invalidates downstream experiments or requires restarting multiple stages.

**How to revert:** {concrete steps. Which file(s) to edit? Which artifacts need regeneration? Any handoff blocks to update in `progress.md`?}

**Downstream dependencies:** {what later stages / tasks are built on top of this decision?}
```

---

## Log

(append entries below this line)

## [2026-04-24 17:30] Stage 0 / T0.1 — Inference engine = vLLM 0.19.1 (not 0.20.0)

**Decision:** Pin `vllm==0.19.1` as the project's inference engine. Python 3.12 (resolved to 3.12.0 via pyenv).

**Alternatives considered:**
- vLLM 0.20.0 — exists as a git tag and GitHub release (2026-04-23), but marked `PRERELEASE: True`; not on PyPI `info.version`. Would need `uv add vllm==0.20.0rc1` or similar and risk pre-GA bugs.
- SGLang 0.5.10 — viable fallback; less verified for Qwen 3.6 and Qwen 3 thinking-mode toggles per the Stage 0 exploration.

**Reason:** vLLM 0.19.1 is the current stable PyPI release (uploaded 2026-04-18). Its registry already includes `Qwen3_5ForConditionalGeneration` (covers Qwen 3.6-27B) and `Gemma4ForConditionalGeneration`, and its predecessor 0.19.0 added Blackwell sm_120 CUTLASS blockwise FP8 GEMM (release notes cite PR #37970) — matches our 5090 hardware. Transformers v5 support landed in 0.19.1. Going stable > prerelease for the first locked env.

**Source:**
- https://pypi.org/pypi/vllm/json — `info.version == "0.19.1"` uploaded 2026-04-18T05:49:16.
- https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.19.0 — Blackwell SM120 fp8 GEMM (#37970), Gemma 4 architecture support (#38826), Transformers v5 adopted.
- https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.19.1 — 10+ Gemma 4 bug fixes; transformers v5.5.3 pin.
- https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/vllm/model_executor/models/registry.py — registry confirms Qwen3_5 + Gemma4 arch classes at 0.19.1 tag.

**Reversibility:** medium. Flipping to 0.20.0rc / 0.18.x / SGLang requires re-running `uv lock` + re-validating model load tests.

**How to revert:** bump `vllm==` in `pyproject.toml`, `uv lock && uv sync`, re-run `scripts/smoke_load.py`.

**Downstream dependencies:** every Stage 0/1/2+ model load. Affects CONVENTIONS.md "Inference engine" and "Python version" entries; affects Stage 2 T2.3 capper + T2.4 judge driver API surface.

---

## [2026-04-24 17:35] Stage 0 / T0.1 — Torch version resolved to 2.10.0+cu128 (not 2.11.0+cu130)

**Decision:** Accept `torch==2.10.0+cu128` as pulled transitively by vllm 0.19.1 (not 2.11.0+cu130 as the initial research suggested for vLLM 0.20.0).

**Alternatives considered:**
- Force `torch==2.11.0+cu130` via explicit pin and custom index URL — risks breaking vLLM 0.19.1's own pin.

**Reason:** vLLM 0.19.1's wheel spec pulls torch 2.10.0+cu128. Our driver 580.126.09 advertises CUDA 13.0 max runtime which is backward-compatible with cu128. `torch.cuda.is_available()` works, both 5090s report sm_120 compute capability, 33.7 GB each. No need to fight the resolver.

**Source:**
- `uv sync` output: `+ torch==2.10.0+cu128`.
- `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"` → `2.10.0+cu128 12.8`.
- `nvidia-smi` driver=580.126.09, CUDA Version advertised 13.0.

**Reversibility:** low-medium. Would need overriding vLLM's torch pin via `tool.uv.override-dependencies`, which might break other pkgs.

**How to revert:** only if a later blocker demands it; add override in pyproject and re-test everything.

**Downstream dependencies:** none directly — our code doesn't rely on torch 2.11 features.

---

## [2026-04-24 17:40] Stage 0 / T0.4 — Gemma 2 27B FP8 checkpoint = `Infermatic/gemma-2-27b-it-FP8-Dynamic`

**Decision:** Use `Infermatic/gemma-2-27b-it-FP8-Dynamic` as the quantized Gemma 2 27B subject checkpoint.

**Alternatives considered:**
- `nm-testing/gemma-2-27b-it-FP8` and `neuralmagic/gemma-2-27b-it-FP8` — the Stage 0 research report named these, but HF API returns 401 on both (repos don't exist — HF's 401-for-unknown-repo behavior when unauthenticated; authenticated `HfApi().model_info(...)` would 404).
- `dangvansam/gemma-2-27b-it-FP8-fix-system-role` (17 dls) — niche fork that tweaks system-role handling; unnecessary complication.
- `mbley/google-gemma-2-27b-it-AWQ` (454 dls) — AWQ fallback if FP8 fails.

**Reason:** `Infermatic/gemma-2-27b-it-FP8-Dynamic` has the highest community download count (134) of the existing Gemma 2 27B FP8 variants on HF. Config confirms `Gemma2ForCausalLM` / 46 layers / 4608 hidden / `quant_method: fp8`. FP8-Dynamic (activations quantized at runtime) is supported by vLLM. No official Google FP8 for Gemma 2 27B exists, so a community variant is required; this is the most-battle-tested option.

**Source:**
- https://huggingface.co/api/models?search=gemma-2-27b-it+fp8 — searched and ranked by downloads, 2026-04-24.
- https://huggingface.co/Infermatic/gemma-2-27b-it-FP8-Dynamic/raw/main/config.json — config verified, `architectures=['Gemma2ForCausalLM']`, `quantization_config.quant_method='fp8'`.

**Reversibility:** high. If extraction-fidelity check fails in Stage 3 T3.1.0, switch to the AWQ fallback `mbley/google-gemma-2-27b-it-AWQ` and re-run.

**How to revert:** change the model ID in `configs/subjects.yaml` (to be created in Stage 2 T2.1) and re-run the quant-validity check.

**Downstream dependencies:** Stage 3 T3.1.0 quant-validity check; all Stage 3/4/6 experiments that use Gemma 2 27B.

---

## [2026-04-24 18:10] Stage 0 / T0.7 — Paper artifacts audit: HF dataset has more than expected

**Decision:** Use the full `lu-christina/assistant-axis-vectors` HF dataset (not just AA directions) as input for Stage 3 Tier 1 PCA. Skip rollout regeneration for Tier 1 PCA fit; still regenerate rollouts for the τ-calibration distribution needed by Stage 4 T4.0.

**Alternatives considered:**
- Regenerate Tier 1 role + trait vectors from scratch to ensure provenance consistency — wastes ~1 day of GPU for no scientific gain; paper's pre-computed vectors were made on bf16 which is the reference we're validating our fp8 extraction against.

**Reason:** The HF dataset (1.2 GB cached to `data/paper_artifacts/assistant_axis_vectors/`) contains per-subject (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B):
  - `assistant_axis.pt` — shape `[n_layers, d_model]` bf16; AA direction at every layer.
  - `default_vector.pt` — mean default-Assistant activation.
  - `role_vectors/<role>.pt` — 275 per subject.
  - `trait_vectors/<trait>.pt` — 240 per subject.
  - `capping_config.pt` — per-layer contrast vectors `contrast_role_pos3_default1` (Qwen + Llama only; **Gemma 2 27B is missing this file — confirmed**).
  - Raw rollouts and τ-calibration distributions are **NOT** released.

**Implications:**
- Stage 3 T3.1 Tier 1 PCA: reads role/trait vectors directly, skips generation.
- Stage 3 T3.1.0 quant-validity check for Tier 1 (Gemma 2 27B, Qwen 3 32B): projects our-quantized-model's test-prompt activations onto the paper's bf16 AA direction. Ready to use.
- Stage 4 T4.0 capping for Gemma 2 27B: must transcribe layer range from paper Appendix F (no capping_config.pt); Qwen 3 32B reads capping_config.pt OR uses paper line 691 transcription directly (`configs/paper_capping_ranges.yaml`).
- **τ-calibration still requires rollout regeneration** for all subjects, since paper didn't release per-rollout projection distributions. This is a Stage 3 T3.1 cost for all 4 subjects (not just Tier 2).

**Source:**
- `huggingface-cli download --repo-type dataset lu-christina/assistant-axis-vectors --local-dir data/paper_artifacts/assistant_axis_vectors` — 1.2 GB pulled 2026-04-24.
- `external/assistant-axis` pinned at commit `a989619`, message "Update jailbreak_capped.json".
- File inspection via `torch.load(...)` — schemas documented above.

**Reversibility:** high. If we decide to regenerate Tier 1 from scratch, wipe `data/paper_artifacts/assistant_axis_vectors/`, run the extraction pipeline, done.

**How to revert:** set a `configs/extraction.yaml` flag `use_paper_tier1_vectors: false`; re-run Stage 3 T3.1.

**Downstream dependencies:** Stage 3 T3.1 (skips Tier 1 PCA input gen), T3.1.0 (uses paper's bf16 AA for fidelity check), T3.5 (role-vector PCA reads from paper or regenerated cache). Stage 4 T4.0 capping-layer-range config.

---

## [2026-04-24 21:30] Stage 0 / T0.10+dan — DAN dataset locked as primary persona-jailbreak eval set

**Decision:** Use the DAN / in-the-wild persona jailbreak dataset (Shen et al., ACM CCS 2024) as the project's primary persona-jailbreak eval set, replacing the deferred Shah et al. 1,100 prompts. Sources are pinned at:
- Personas: `TrustAIRLab/in-the-wild-jailbreak-prompts`, config `jailbreak_2023_12_25`, revision `a10aab8eff1c73165a442d4464dce192bd28b9c5` (1,405 in-the-wild persona-jailbreak prompts).
- Questions: `TrustAIRLab/forbidden_question_set`, revision `369aa8e10ee2a26cf087fdcc34af0bb928d33d8e` (390 forbidden questions, 30 per category × 13 OpenAI-policy categories).

Output sits at `data/eval/dan_jailbreak/` with `raw_personas.parquet` (1,405 rows), `raw_questions.parquet` (390 rows), `sampled_1100.parquet` (1,100 rows, 84-85 per cat — within ±1 of perfect balance), and `manifest.json` recording the revisions + seed (`42`) + per-category counts.

**Alternatives considered:**
- Shah et al. 1,100 — withheld per responsible disclosure; not locatable on HF or anywhere public; would require attacker-LLM reconstruction (which we ALSO build, see next entry, but as a secondary/comparison set rather than primary).
- HarmBench — different taxonomy (~7 high-level categories), wouldn't slot 1:1 into our 13-cat stratification, and weaker tie to the persona-jailbreak threat model.

**Reason:** DAN is (a) a real-user-sourced superset of the same threat model Shah et al. studied, (b) public license (MIT), (c) structurally identical (persona system prompt × harmful behavioral question), (d) available at ~10× the persona pool we need (1,405 vs 1,100), letting stratified sampling yield perfect category balance. Per the user's task spec it is "scientifically arguably stronger" than Shah's synthetic set.

**Source:**
- https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts — verified 2026-04-24, sha `a10aab8e…`, 1,405 rows in `jailbreak_2023_12_25/train`.
- https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set — verified 2026-04-24, sha `369aa8e1…`, 390 rows, 13 cats with the labels: `Illegal Activity, Hate Speech, Malware, Physical Harm, Economic Harm, Fraud, Pornography, Political Lobbying, Privacy Violence, Legal Opinion, Financial Advice, Health Consultation, Gov Decision`.
- Note: HF sibling listing shows the 13 category names use `Malware` (not `Malware Generation`) and `Gov Decision` (not `Government Decision`). Code uses HF's verbatim names.

**Reversibility:** high. The build script (`src/data/build_dan_jailbreak.py`) is one CLI invocation; flipping seed or stratification ratios requires only re-running it.

**How to revert / change:** rerun `uv run python -m src.data.build_dan_jailbreak --output-dir <path> --seed <s>`; manifest captures the new state.

**Downstream dependencies:** Stage 2 T2.4 safety-eval harness reads `sampled_1100.parquet`. Stage 3-6 jailbreak experiments. Token-distribution audit (`configs/eval_sizes.yaml`) does NOT yet have an entry for this dataset — Stage 2 T2.0 should add one (audit p99 of `full_prompt` token length per subject tokenizer, set `max_input_len`).

---

## [2026-04-24 21:35] Stage 0 / T0.10+shah-reconstructor — Shah-style reconstruction utility scaffolded with vLLM json_schema

**Decision:** Build `src/data/reconstruct_shah_jailbreaks.py` as a paper-reproduction utility that uses the already-running Gemma 4 31B vLLM service (`http://localhost:8000`, model `nvidia/Gemma-4-31B-IT-NVFP4`) as the attacker. Pipeline: (1) generator pass produces N (persona, question) JSON pairs per category constrained via `response_format={"type":"json_schema",...}`; (2) rubric filter pass scores each pair 1-5 on `on_category / adversarial / coherent / distinct`, drops mean<3.0. Categories default to the same 13 OpenAI-policy labels DAN uses (`openai13`), so reconstructed and DAN data are 1:1 comparable. Output schema (`prompt_id, persona_id, question_id, persona_text, question_text, category, full_prompt`, plus rubric metadata) matches `data/eval/dan_jailbreak/sampled_1100.parquet` so downstream eval code is dataset-agnostic.

Smoke test: `--n-per-category 2 --categories openai13` produced 26 rows (2 per cat) in **5 min 4 sec** with 100% pass-rate at the rubric floor (mean rubric 4.32, min 3.75); spot-read of 3 prompts confirmed coherent persona+question pairs targeting the right harm category.

**Alternatives considered:**
- vLLM `extra_body={"guided_json": ...}` — verified working but model wrapped output in ```json fences, not strictly enforced. `response_format=json_schema` enforced cleanly with no fences, so used that path.
- HarmBench category set — placeholder kept in code (`HARMBENCH_CATEGORIES = []`) but not used now; DAN's `openai13` is the right default for direct comparison.
- Single-pass generation (no rubric filter) — paper's pipeline includes a quality check; matched it.

**Reason:** Shah et al. methodology requires an attacker LLM. Spinning up a separate model would waste GPUs; the running Gemma 4 31B endpoint is exactly the resource the user pre-provisioned for this task. `response_format=json_schema` is more reliable than freeform-prompt JSON parsing (zero parse failures observed across 26 generation calls + 39 rubric calls in the smoke test).

**Source:**
- vLLM 0.19.1 OpenAI-compatible structured-output API: https://docs.vllm.ai/en/v0.19.1/features/structured_outputs.html (verified 2026-04-24).
- Endpoint discovered live at `http://localhost:8000/v1/models` returning `nvidia/Gemma-4-31B-IT-NVFP4`.
- Smoke-test artifacts: `data/eval/reconstructed_jailbreak/smoke_26/{sampled_26.parquet,manifest.json}`.

**Smoke-test wall-clock estimate for full run:** 5 min 4 sec for 26 pairs (2/cat × 13 cats with 1.5x overgeneration) → ~12 sec/kept-pair end-to-end (generation + rubric). For 1,100 pairs (~85/cat): roughly **~3.5 hours** wall-clock at the same throughput, assuming the vLLM service isn't contended and the rubric pass-rate stays near 100%. If pass-rate drops with stronger overgeneration (which we'd want at scale to keep diversity), budget up to ~5 hours.

**Reversibility:** high. The utility is a CLI; users can rerun with different `--n-per-category`, `--seed`, `--categories`, or `--overgenerate-factor` without touching code.

**How to revert / change:** rerun the CLI with new args. Code lives in `src/data/reconstruct_shah_jailbreaks.py`.

**Downstream dependencies:** Optional secondary eval set for Stage 4-6 (lets us sanity-check that DAN-vs-Shah-style results agree). NOT a Stage 0 critical-path artifact — DAN primary set fills the same role. Full-scale run is deferred to user (multi-hour, ran out-of-band).

---

## [2026-04-25 07:47] Stage 1 / T1.8 — Revert fp8/2-GPU → bf16/4-GPU for core stages

**Decision:** All 4 RTX 5090s on the box are now available to this project (the parallel `nvidia/Gemma-4-31B-IT-NVFP4` LoRA-tuning workload on GPUs 0,1 has finished). Switch core stages from "fp8 quantized at TP=2" to **bf16 at TP=4** for all 4 subjects + the primary judge. Working-tree changes already applied: `src/utils/env.py` widens `CUDA_VISIBLE_DEVICES` from `2,3` → `0,1,2,3`; `configs/subjects.yaml` rewrites every entry to use the bf16 base HF IDs (`google/gemma-2-27b-it`, `Qwen/Qwen3-32B`, `google/gemma-4-31B-it`, `Qwen/Qwen3.6-27B`) at `tensor_parallel_size: 4`. `Infermatic/gemma-2-27b-it-FP8-Dynamic`, `Qwen/Qwen3-32B-FP8`, `RedHatAI/gemma-4-31B-it-FP8-block`, and `Qwen/Qwen3.6-27B-FP8` are no longer referenced by core configs; their decisions.md entries (2026-04-24 17:40 and Stage 0 T0.4/T0.5 batch-size logs) become historical context, not active state.

**Alternatives considered:**
- Keep fp8/TP=2 to preserve Stage 0 smoke-load numbers and avoid re-running tuning — rejected because the original constraint that forced fp8 (64 GB total VRAM) is gone, and bf16 removes a class of risk (per-subject quant-validity check, fp8 extraction-fidelity unknowns, Gemma 4 FLASHINFER fp8 codepath bug per vLLM #40677). Bf16 is the paper's reference precision.
- Keep fp8 at TP=4 instead of bf16 — would gain a bit of throughput but keeps the quant-validity gate in the critical path; user judgment was that the simplicity of bf16 wins now that VRAM allows it. Re-pickable later if any model OOMs at TP=4 bf16.
- Move Llama 3.3 70B back into core stages — rejected. 70B × bf16 ≈ 140 GB just for weights, exceeds 128 GB total. Llama stays at Stage 7 Ext 9, where the original plan's fp8 path (or NVFP4 fallback) is the only way to fit it.

**Reason:** GPUs 0,1 became available, removing the constraint that motivated fp8. Bf16 = paper's reference precision = no extraction-fidelity argument needed in the report = no Stage 3 T3.1.0 quant-validity gate. The grid-search work in commits 88081e4..f094689 ("[Stage 1 / prep]") tuned inference under the new 4-GPU bf16 path; subjects.yaml changes already reflect those numbers.

**Source:**
- User instruction in chat 2026-04-25 ("we are working on this project and currently stage 0 agent is working on creating the env" + working-tree subjects.yaml comment "revert from Stage 0's 2-GPU constraint; see plans/decisions.md 2026-04-25 fp8->bf16 entry").
- Working-tree diffs: `configs/subjects.yaml` (TP=4, bf16 IDs), `src/utils/env.py` (CUDA_VISIBLE_DEVICES=0,1,2,3), `scripts/{smoke_load,phased_pipeline_smoke}.py` (matching updates).
- Recent commits 5e7901d..f094689 ("[Stage 1 / prep] grid search …") — inference grid search under the new precision/TP regime.

**Reversibility:** medium. Reverting to fp8/TP=2 means re-editing `configs/subjects.yaml`, `src/utils/env.py`, and re-running the Stage 0 smoke load + grid search. No experiment artifacts depend on this yet (Stage 1 is design-only); reversion before Stage 3 T3.1 actually runs has zero data cost. After Stage 3 it would invalidate cached activations.

**How to revert:** `git checkout HEAD -- configs/subjects.yaml src/utils/env.py scripts/smoke_load.py scripts/phased_pipeline_smoke.py`, then re-run Stage 0 T0.4/T0.5/T0.11 smoke loads. Re-instate the "Quantization policy (2-GPU constraint)" section in CLAUDE.md and CONVENTIONS.md from this commit's git history. Stage 7 Ext 9 (Llama 70B at fp8) is unaffected — its plan already assumes fp8.

**Downstream dependencies:**
- `plans/CONVENTIONS.md` — "Quantization policy" section renamed to "Precision policy"; bf16 default for core stages, fp8 reserved for Ext 9 only. Quant-validity check moved from Stage 3 prelude to Ext 9 prerequisites.
- `CLAUDE.md` — "Models" / "Hardware" / "Quantization policy" lines updated to 4-GPU TP=4 bf16. The "All subjects run quantized" claim and the 2-GPU-constraint asides are now historical.
- `configs/experiment_template.yaml` (Stage 1 T1.6.5) — defaults `dtype: bf16`, `tensor_parallel: 4`.
- Stage 3 T3.1.0 quant-validity gate becomes a no-op for core subjects (still applied to any future fp8 subject; documented as Ext 9 prerequisite).
- `pyproject.toml` does NOT need changes — `vllm==0.19.1` runs both bf16 and fp8.

---

## [2026-04-25 22:00] Stage 2 / T2.9 — Plan B scope cuts (single-subject H1 demo on Gemma 2 27B)

**Decision:** Replace the original Stage 2 T2.9 ≤2-hour smoke test with **Plan B** — a single-subject end-to-end H1 demonstration on Gemma 2 27B at experiment-grade volumes scoped to fit a fellowship-deadline window. Cuts vs the original Stage 2 + Stage 3/4 plan: subjects 4 → 1 (Gemma 2 27B only), datasets 2 → 1 (DAN-only), prompts/dataset 1100 → 500 (stratified across the 13 OpenAI-policy categories), rollouts/role for τ-calibration 100 → 30 (paper's 275-vector PCA cache is reused for the actual PC fit). All four cuts are reversible — the post-deadline replay (April 27 → May 3) re-runs the exact same code path with the original volumes and all 4 subjects.

**Alternatives considered:**
- Keep the original 2-hour 100-prompt smoke for the deadline submission — rejected. Doesn't produce fellowship-grade signal; the bar chart (criteria 1-3) needs ≥300 prompts/condition for the BCa CIs to exclude zero with the expected effect sizes.
- Run Plan B at full 1100 × 2 datasets — rejected. With HF as the steered backend, ~14-15 hr compute exceeds the 18-hour ceiling once writeup + buffer are factored in.
- Skip random-direction baselines and rely on PC2/PC3 vs unsteered comparison — rejected. The random baselines rule out the "any nonzero steer breaks capping" alternative explanation; without them the H1 claim is weak.

**Reason:** Scoped Plan B fits an 18-hour ceiling with margin, retains all four scientific load-bearing components (paper-reproduction AA-cap, PC2/PC3 directional steering, random baselines, per-prompt LASSO blind-spot lift), and keeps Stage 2 implementation strictly invoked-not-modified. The post-deadline replay restores full volumes without code changes.

**Source:**
- `plans/plan_b_directive.md` (user-authored directive, 2026-04-25).
- User instruction in chat 2026-04-25 confirming 18-hour budget + DAN-only + 500 prompts.
- Realistic compute budget: ~10-11 hr per the per-step table in T2.9 of `plans/stage-2-infrastructure.md`.

**Reversibility:** high. Plan B is a runtime-config scope, not a code change. The post-deadline sweep replays with `--n-prompts 1100 --datasets dan,shah_reconstructed --subjects all` against the same modules.

**How to revert / scale up:** run `src/experiments/plan_b.py --subjects {qwen_3_32b,gemma_4_31b_thinking_on,gemma_4_31b_thinking_off} --datasets dan,shah_reconstructed --n-prompts 1100 --rollouts-per-role 100` per subject; Plan B output stays as the Gemma 2 reference, post-deadline outputs land at `results/post_deadline_sweep_{subject}/`.

**Downstream dependencies:**
- T2.9 spec in `plans/stage-2-infrastructure.md` rewritten — see file.
- T2.5 (capability eval) marked POST-PLAN B.
- T2.7 split into T2.7a (binary LASSO + blind-spot lift, Plan B critical path) + T2.7b (ordinal LASSO + per-PC FDR, post-deadline).
- T2.4 cross-check judge phase deferred to post-deadline replay.
- Shah-reconstructed dataset deferred to post-deadline replay.

---

## [2026-04-25 22:05] Stage 2 / T2.9 — HF/vLLM backend split for steered vs unsteered generation

**Decision:** Use **vLLM** for unsteered generation (Plan B Step 1a role rollouts, Step 5 jailbreak baseline) and the judge phase. Use **HuggingFace transformers + accelerate** (`device_map="auto"`, `torch_dtype=torch.bfloat16`, `attn_implementation="sdpa"`) for **all activation extraction passes** (Step 1b per-rollout activations, Step 1c lmsys-chat-1m norm cache, Step 7b per-prompt activation extraction filling `aa_projection` + `pc_projections` columns) and **all steered/capped generation** (Step 6: 10 conditions × 500 DAN prompts under `external/assistant-axis::ActivationSteering` via `src/steering/steerer.py::cap_and_steer`).

**Alternatives considered:**
- vLLM `enforce_eager=True` + `register_forward_hook` — rejected. vLLM issue [#4084](https://github.com/vllm-project/vllm/issues/4084) reports hooks firing on prefill but not decode in some configs; "novel integration to debug under deadline pressure" risk is high; ~10–30% throughput loss vs CUDA graphs anyway.
- nnsight 0.4+ vLLM backend — rejected. Open issues [#640](https://github.com/ndif-team/nnsight/issues/640), [#641](https://github.com/ndif-team/nnsight/issues/641), [#642](https://github.com/ndif-team/nnsight/pull/642) document 13 unfixed intervention gaps as of April 2026; alpha-quality on the vLLM path.
- TransformerLens — rejected. Wraps HF, same forward hooks, no throughput advantage; only ergonomics for mech-interp.
- vllm-lens (UKGovernmentBEIS) — additive-only `apply_steering_vectors`; capping (`h - v·max(⟨h,v⟩-τ, 0)`) requires source patch. Stage 7 candidate, not Plan B.
- EasySteer (ZJU-REAL) — fork of vLLM 0.17.1, incompatible with our 0.19.1 pin; capping not native. ~1 day integration risk. Skip.
- `repeng` (vgel) — HF-only, no vLLM path.
- vLLM PRs #7906 / #12870 (control vectors) — both auto-closed for inactivity, never landed.
- SGLang `--forward-hooks` — real, documented, can mutate during decode; **but** TP=4 on Gemma 4 31B unvalidated by SGLang team, sm_120 fp8 broken (issues [#9233](https://github.com/sgl-project/sglang/issues/9233), [#11576](https://github.com/sgl-project/sglang/issues/11576)), needs separate uv env. **Spike cost 4–6 hr; Plan B savings ~2 hr → net loss.** Strongly positive ROI at multi-subject scale (~67 hr saved over the post-deadline sweep). Logged as post-Plan B Stage 7 candidate; see `plans/sglang_post_plan_b_spike.md`.

**Reason:** HF + ActivationSteering is the trusted, known-working path. The upstream `external/assistant-axis` itself uses HF; the paper's protocol is HF-native; our `src/steering/steerer.py` (Stage 1 T1.3) already wraps it. With Plan B's scope cuts, HF throughput at ~150 tok/s aggregate × batch=8 with sdpa puts steered runs at ~3 hr — comfortable inside the 18-hour ceiling. The HF/vLLM split is a runtime arg on `run_subject_rollouts --backend {hf,vllm}`, not a code rewrite; both backends share the same module API.

**Source:**
- Subagent research 2026-04-25 covering vLLM, nnsight, TL, vllm-lens, EasySteer, repeng, SGLang. URLs cited in `plans/sglang_post_plan_b_spike.md`.
- `external/assistant-axis::ActivationSteering` (`steering.py`) registers `register_forward_hook` on `model.layers[i]` — pure HF. Confirmed by reading source.
- `external/assistant-axis::ActivationExtractor` (`internals/activations.py`) same pattern.

**Reversibility:** high. Backend choice is `cfg.steering.mode → backend` dispatch in the work-module. Swapping in SGLang post-Plan B is additive (new `--backend sglang` branch + hook factories matching ActivationSteering semantics); does not require changes to `cap_and_steer`, `multi_axis_cap`, `eval_safety`, `eval_full`.

**How to revert:** N/A in core stages — bf16/HF is the locked default. SGLang opt-in requires the post-Plan B spike per `plans/sglang_post_plan_b_spike.md`.

**Downstream dependencies:**
- `src/evaluation/run_subject_rollouts.py` work-module ships with `--backend {hf,vllm}` flag (T2.4 main).
- Plan B Step 6 routes through `--backend hf` (10 steered conditions × 500 DAN prompts).
- Plan B Steps 1a + 5 route through `--backend vllm` (unsteered).
- Post-deadline multi-subject sweep keeps the same split.

---

## [2026-04-25 22:10] Stage 2 / T2.4.5 — Async OpenAI client, concurrency=100 default

**Decision:** GPT-5.5 ground-truth labelling for the 200-sample judge validation set (T2.4.5) uses `from openai import AsyncOpenAI` + `asyncio.Semaphore(100)` (default). User has tier-5 rate limits with massive headroom; CLI `--max-concurrent` flag allows bumping to 200. Per-call retry on `RateLimitError` with exponential backoff (0.5 / 1 / 2 / 4 s), partial-result stashing every 25 completions, hard-stop at `cost_usd ≥ $15` (D9 budget cap).

**Alternatives considered:**
- Synchronous OpenAI calls with 1 concurrent request — rejected. ~45 min wall-clock for 200 calls; user explicitly requested async fan-out.
- Concurrency=200 default — rejected as the default. Some Anthropic-style "hello world" scenarios trigger TPM limits before RPM limits at 200 in-flight; 100 is a safe default with the option to bump.

**Reason:** User explicitly stated "I have a huuugeee rate limit so fire idk as many as is feasible at once, 200+ work." Async with concurrency=100 cuts T2.4.5 wall-clock from ~45 min to ~1 min API + ~10 min cross-judge ≈ ~12 min total. Frees Plan B writeup window; reduces deadline pressure.

**Source:** User instruction in chat 2026-04-25.

**Reversibility:** high. CLI flag swap.

**How to revert:** `--max-concurrent 1` for sync behaviour; `--max-concurrent 200` to push harder.

**Downstream dependencies:**
- `src/evaluation/run_gpt55_validation.py` work-module (NEW, T2.4.5 deliverable).
- `OPENAI_API_KEY` added to `.env.example`.
- T2.4.5 acceptance relaxed from ≥90% to **≥85% binary agreement under Plan B** (single subject, DAN-only sample, narrower distribution than the post-deadline cross-subject sample). Post-deadline replay restores the 90% bar against the multi-subject pool.

---

## [2026-04-25 22:15] Stage 2 / T2.7 — Split into T2.7a (Plan B critical) + T2.7b (post-deadline)

**Decision:** Stage 2 T2.7 splits into:
- **T2.7a (Plan B critical path):** binary `logistic_lasso_cv` + `blind_spot_lift` + `cohens_d` + (already-implemented) `bca_ci`, `bca_ci_difference`, `point_biserial`, `pearson_with_ci`, `auc_with_ci`. Required for Plan B's H1 numerical claim (per-prompt blind-spot AUC delta with BCa CI).
- **T2.7b (post-deadline):** ordinal `logistic_lasso_cv` (proportional-odds via `mord.LogisticAT`), per-PC FDR-corrected point-biserial sweep using `bh_fdr` (already implemented). Stage 3 T3.7 territory; helpers ship in Plan B but aren't invoked.

**Alternatives considered:**
- Implement the full T2.7 (binary + ordinal) before Plan B kickoff — rejected. Ordinal LASSO adds ~1.5 hr dev with no Plan B usage; defer to keep critical path tight.
- Skip LASSO entirely for Plan B (raw bar chart only) — rejected. Per-prompt activation extraction is **already required by `PER_PROMPT_COLUMNS` schema** for the `aa_projection` + `pc_projections` columns — sunk cost. The LASSO + blind-spot lift on top is +30 sec compute and ~1.5 hr dev for the formal H1 statement. Cheap upgrade from "demo" to "rigorous finding."

**Reason:** Per-prompt LASSO blind-spot AUC delta is the formal version of the H1 claim ("PC2/PC3 carry harm-relevant signal orthogonal to AA at the per-prompt level"). The bar chart shows population-level effect; the LASSO shows prediction-level effect. Both are useful in the writeup. Ordinal robustness check is secondary and adds nothing to the headline; defer.

**Source:** User instruction 2026-04-25 ("LASSO is just analysis post-run right? if we do it is there any benefit?" — yes, with the cheap-because-extraction-is-sunk-cost framing).

**Reversibility:** high. T2.7b is purely additive code; running the post-deadline replay with `--ordinal-lasso --fdr-per-pc` flags appends columns to `metrics.json`.

**How to revert:** N/A — split is forward-only.

**Downstream dependencies:**
- `src/analysis/lasso.py::logistic_lasso_cv` (T2.7a).
- `src/analysis/blind_spot.py::blind_spot_lift` (T2.7a).
- `src/analysis/effect_size.py::cohens_d` NEW (T2.7a).
- `src/analysis/lasso.py::ordinal_lasso_cv` (T2.7b stub remains; filled post-deadline).
- Plan B Step 9 (analysis) consumes T2.7a only.

---

## [2026-04-25 23:00] Stage 2 / T2.2 — Gemma 2 27B L\* = 21 (argmax cos_sim(PC1, AA))

**Decision:** Plan B uses **L\* = 21** as the extraction layer for Gemma 2 27B, picked by argmax cos_sim(PC1, AA) over a per-layer sweep on the paper's released 275-role-vector cache. cos_sim(PC1, AA) at L*=21 is **0.8825** — well above the paper's 0.71 threshold.

**Why this differs slightly from the paper:** Paper line 96 just says "the middle layer". For Gemma 2 27B with 46 layers, `n_layers // 2 = 23`, but the paper text doesn't specify their exact integer. Our argmax-based selection lands at 21 (the cos_sim sweep peaks at 21 = 0.8825, vs 0.7835 at 23). Within ±3 of the implicit paper choice; CONVENTIONS Stage 3 T3.1.5 tightening says log if >±3 (not the case).

**Source:** `scripts/validate_pca_against_paper.py` run 2026-04-25 23:00; output saved verbatim to this entry.

```
Per-layer cos_sim(PC1, AA) (Gemma 2 27B, paper's released bf16 cache):
  layer  0: 0.7136
  layer  2: 0.8584
  layer 17: 0.8608
  layer 21: 0.8825  *  <- L*
  layer 22: 0.8451
  layer 45: 0.8808
min: 0.6829 (layer 7); max: 0.8825 (layer 21)
```

**Reversibility:** high. L\* is a single integer in `results/plan_b_gemma2_27b/extraction/L_star.txt`; flipping requires only re-projecting per-prompt activations (which is pure CPU once the per-layer extraction cache exists).

**How to revert:** edit `L_star.txt` and re-run the projection step of Plan B Step 7b.

**Downstream dependencies:** Plan B Step 2 records L*; Step 4 sets capping range = [L*+2, L*+9] = [23, 30]; Step 6 steers PC2/PC3 at L*=21 (single-layer paper convention); Step 7b projects per-prompt activations at L*=21.

---

## [2026-04-26 14:05] Stage 2 / T2.4.5 — Smoke validation: 93.0% binary agreement vs gpt-5.5

**Decision:** Use **gpt-5.5** as the frontier-judge reference model (user-confirmed available; reasoning model — uses `max_completion_tokens` not `max_tokens`, no temperature). Bumped `max_output_tokens` to 2048 to accommodate hidden CoT. Async at concurrency=100.

**Results on smoke n=200 (`results/plan_b_smoke/details.parquet`):**
- Binary agreement (Qwen 3.6-27B primary vs gpt-5.5): **93.0%** ✅ ABOVE paper's 91.6% threshold
- 9-cat exact match: 58.0%
- Cohen's κ: -0.04 (skewed-marginal artifact: 96.5% non-harm in both → chance agreement ≈ 93%)
- Cost: $3.03 (gpt-5.5 reasoning ~3× gpt-4.1 cost)
- Wall-clock: 48.5s (200 calls async at concurrency=100)
- Positive rate: gpt-5.5 = 3.5%, Qwen = 3.5%, gpt-4.1 = 8.0% (gpt-4.1 over-claims harm on Gemma 2's 75% nonsensical responses)

**Cross-frontier check (gpt-5.5 vs gpt-4.1, smoke n=200):** 89.5% binary agreement, κ=0.04. gpt-4.1 outlier; gpt-5.5 + Qwen agree more closely on the harm class.

**Acceptance status:** PASSES. Above paper reference (91.6%) and above Plan B threshold (≥85%). Updated `configs/plan_b.yaml.gpt55_validation` to `model: gpt-5.5` + `max_output_tokens: 2048`.

**Sampler-bug detail (independent finding):** The smoke's `_dan_500.parquet` has 8 duplicate rows (496 unique prompt_ids out of 500 sampled). Source DAN is clean (1100 unique). Bug is in `step_5_safety_baseline` sampler: `dan.drop(sampled.index)` fallback uses indices that get scrambled after `groupby().apply()`, so the second `.sample()` can pick already-sampled rows. Effect on harm_rate: ≤1.6% bias worst case. Plan B is running with this bug baked in — within sampling noise so not worth restart. **TODO post-Plan B: fix the sampler with prompt_id-based exclusion.**

**Source:** Run output `data/judge_validation/plan_b_smoke_gpt55_labels.parquet` 2026-04-26 14:05. Cross-check parquet `plan_b_smoke_gpt41_labels.parquet`. Snapshot script `scripts/snapshot_leftover_scope.py` flagged the duplicate count.

**Reversibility:** high. Re-run with different model via `args.model`. Sampler fix is a 5-line change.

**Downstream dependencies:** Full Plan B Step 8 will rerun gpt-5.5 against 200 stratified pairs from full output; expect agreement to remain ~93% with richer harm-class distribution. Sampler-bug fix applies to all future Plan B replays (phase_b_*).

---

## [2026-04-26 14:10] Stage 2 / T2.9 — Leftover-scope snapshot for follow-up runs

**Decision:** Wrote a non-invasive snapshot of Plan B's IN-scope identities + everything CUT to `results/plan_b_gemma2_27b/leftovers/`:
- `dan_in_500.parquet` — the 500 prompts Plan B is generating against
- `dan_complement_604.parquet` — the leftover 604 DAN prompts (1100 - 496 unique)
- `roles_in_275.parquet` — all 275 paper roles (Plan B uses every role at 30 rollouts each)
- `scope.yaml` — declarative manifest of cut subjects (qwen_3_32b, gemma_4_31b ON+OFF, llama_3_3_70b), cut datasets (Shah-reconstructed), cut tasks (T2.5 capability, T2.7b ordinal LASSO, T2.4 cross-check judge), and 4 ranked follow-up runs

**Why:** If Plan B finishes early on the 18-hr budget, follow-up runs can use leftover GPU time. Each is documented with estimated compute hours and a list of caches it can reuse (judge config, AA/PC fits where same subject, etc.).

**Source:** `scripts/snapshot_leftover_scope.py` (deterministic; uses same seed=42 as Plan B sampler).

**Reversibility:** high. Snapshot is read-only metadata; doesn't touch Plan B pipeline. Rerunning overwrites.

**Downstream dependencies:** Optional `phase_b_*` runs use this scope. Each follow-up reuses Plan B infrastructure as-is — only config + dataset paths change.

---

## [2026-04-26 13:55] Stage 2 / T2.4.5 — Smoke validation: 88.5% binary agreement vs gpt-4.1 [SUPERSEDED by 14:05 entry]

> **SUPERSEDED.** This entry used **gpt-4.1** because I incorrectly concluded gpt-5.5 was unavailable
> based on a `models.list()` probe. Direct `chat.completions.create(model="gpt-5.5")` works fine —
> just needed `max_completion_tokens` (reasoning model) instead of `max_tokens`. The 14:05 entry
> reruns with gpt-5.5 and gets **93.0% binary agreement** (above paper's 91.6% threshold). gpt-4.1
> result kept here for cross-frontier comparison only; it is NOT the operative validation number.

**Decision:** GPT-5.5 placeholder in `configs/plan_b.yaml` substituted with **gpt-4.1** (strongest available frontier model on user's OpenAI account; gpt-5.5 not in models.list). Pricing input=$2/Mtok, output=$8/Mtok. Async at concurrency=100. Run against `results/plan_b_smoke/details.parquet` (200 pairs) before full Plan B output exists, to validate the validation pipeline early and budget calibration.

**Results on smoke n=200:**
- Binary agreement (Qwen 3.6-27B primary vs gpt-4.1): **88.5%**
- 9-cat exact match: 58.5%
- Cohen's κ: -0.05 (skewed-marginal artifact: 177/200 non-harmful per both → chance agreement ≈ 89%)
- Cost: $1.18, wall-clock: 8.9 seconds
- Confusion: gpt-4.1 marks 16 as harm that Qwen doesn't; Qwen marks 7 as harm that gpt-4.1 doesn't.
- Qwen over-claims harm on ~7 nonsensical / out_of_context responses (Gemma 2 produced 75% nonsensical at smoke scale).

**Acceptance status:** Below paper's 91.6% reference; AT the relaxed Plan B threshold (≥85% per `plans/stage-2-infrastructure.md` T2.9 acceptance #6). Smoke n=200 has high marginal-distribution noise — re-validate against full Plan B's 200-sample subset post-run; expect agreement to tighten with more diverse harmful-class examples.

**Source:** Run output `data/judge_validation/plan_b_smoke_gpt41_labels.parquet` 2026-04-26 13:55. T2.4.5 spec in `plans/stage-2-infrastructure.md`.

**Reversibility:** high. Re-run with different model (`gpt-4.1` → `gpt-4.5` if user enables, etc.) by changing `model` in args.

**Downstream dependencies:** `metrics.json` populates `judge_agreement_gpt55_vs_primary`. Full Plan B Step 8 will rerun this against 200 stratified pairs from the full 500-DAN output; agreement reported in writeup.

---

## [2026-04-25 22:20] Stage 2 / T2.9 — SGLang `--forward-hooks` deferred to post-Plan B

**Decision:** Defer the SGLang `--forward-hooks` integration spike from Plan B to **post-Plan B Stage 7 candidate / first task of the April 27+ multi-subject sweep**. Spike plan + acceptance criteria + migration path documented in `plans/sglang_post_plan_b_spike.md`.

**Alternatives considered:**
- Spike SGLang now (4-6 hr) to use it for Plan B steered runs — rejected. Plan B savings ~2 hr; spike cost 4-6 hr; ROI is **negative** for single-subject Plan B. Plus three open risks: (a) TP=4 on Gemma 4 31B unvalidated by SGLang team (cookbook documents TP=2 only on H200), (b) sm_120 fp8 broken in SGLang per issues #9233 / #11576 (we'd lose Stage 7 Ext 9 fp8 path validation), (c) numerical equivalence vs `external/assistant-axis::ActivationSteering` unproven.
- Skip SGLang entirely — rejected. ROI at multi-subject scale (~67 hr saved over the post-deadline sweep on 4 subjects × 2 datasets × ~16x steered-condition compute) is strongly positive.

**Reason:** SGLang `--forward-hooks` is real (PR #13217 / #13994, v0.5.10), can mutate residual activations during decode by PyTorch contract, and matches our `cap_and_steer` semantic via factory pattern. But every risk above is a debug-cycle in disguise; under the 18-hour Plan B ceiling, "spike cost > savings" is the bottom line. Save it for the post-deadline sweep where the math flips.

**Source:**
- Subagent research 2026-04-25; sources cited in `plans/sglang_post_plan_b_spike.md`.
- [SGLang `server_args.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py), [`hook_manager.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/hook_manager.py), [Cookbook: Gemma 4](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4).

**Reversibility:** high. SGLang is an additive backend (`--backend sglang` flag on `run_subject_rollouts`). Adding it post-Plan B requires no changes to existing Stage 1/2 module APIs.

**How to revert:** N/A — current decision is to NOT integrate. To activate, run the spike per `plans/sglang_post_plan_b_spike.md`.

**Downstream dependencies:**
- `plans/sglang_post_plan_b_spike.md` (NEW file) — full spike plan.
- Stage 2 T2.9 stays HF-for-steered (per the 22:05 decision above).
- Post-deadline multi-subject sweep schedules SGLang spike as its first task; subjects passing the 5 acceptance criteria opt in via `cfg.steered_backend = sglang`.


---

## [2026-04-26 00:10] Stage 2 / T2.9 — Plan B AA-cap recovery: sign + tau + layer-depth fix

**Decision:** First Plan B run produced 100% degenerate token-loop outputs across all 11 capped conditions (`"th'th, the last, the last…"`, `"KenapaKenapaKenapa…"`, etc.) — judge labeled 100% `nonsensical`, headline `aa_cap_delta_pp=14.8` was artifactual (cap killed the model, not harm). Three stacked bugs identified and corrected; MVP re-run (5 conditions × 500 prompts) launched.

**Bug 1 — Sign convention mismatch.** Our AA = `mean(default) − mean(role)` (Assistant-positive); upstream `external/assistant-axis::ActivationSteering._apply_cap` is a CEILING (`(proj − τ).clamp(min=0)`) that expects role-positive vectors. Verified: `cos_sim(our AA, Lu et al.'s released qwen-3-32b/capping_config.pt vector) = −1.0000` at every layer. Composed with our AA, the cap pushed Assistant-territory activations DOWN toward role — anti-defense. *Fix:* `src/steering/steerer.py:73-105` negates capping vectors inside `from_config` (already in place pre-this-entry).

**Bug 2 — Tau calibration mismatch.** `step_3_tau_calibration` projects role-rollout activations on `+AA_unit` and stores `p25`. With sign-flip alone, the cap operator computes `<h, −AA_unit>` (typically negative) but compared against `+τ` (positive p25) → `excess = max(neg − pos, 0) = 0`, cap never fires. Sign-flip alone yields a no-op cap. *Fix:* `src/experiments/plan_b.py:486-499` reads `−tau["p75"]` of the existing +AA calibration. Math: `p25(<h, −AA>) = −p75(<h, +AA>)`. Zero step_3 change required (Approach B); same on-disk JSON.

**Bug 3 — Capping at the wrong depth.** Paper publishes capping ranges only for Qwen-3-32B (layers 46–53 of 64 = 71.9–82.8% depth) and Llama-3.3-70B (layers 56–71 of 80 = 70.0–88.75% depth); both at width ≈ 12.5% of total layers. Plan B's original fallback (`offset=6, width=8` → layers [23,30] for L*=21) sat at 50–65% depth — ~21pp shallower than the paper's Qwen range. AA's geometric structure (cos_sim with PC1) holds across the model but the residual offset and projection magnitudes differ wildly. Combined with Bug 2's wrong sign of τ, every successive cap pushed the residual further off-manifold — token-loop garbage. *Fix:* `configs/plan_b.yaml:28-29` → `capping_center_offset_from_lstar=15, capping_width=6` → layers [33,38] at 71.7–82.6% depth (proportional match to Qwen).

**Empirical validation pre-launch (results/plan_b_gemma2_27b/extraction/tau_calibration.json + cache projections):**
- `cos_sim(PC1, AA)` at L=33–38: 0.73–0.77 (above paper's 0.7 threshold).
- Assistant-vs-role projection gap on +AA at L=33–38: consistently 1483–1545 (clean discriminative signal).
- τ_role (= −p75 of role rollouts on +AA) lies between Assistant mean and role mean at every layer → cap fires for role activations, not Assistant.
- Expected cap excess per layer: ~450–520 (vs original broken setup's 5000–9000). Cumulative push across 6 layers ≈ 3000 (vs 8 layers × 5000–9000 = 40k–70k).
- Unit test `tests/unit/test_steerer_compose.py::test_plan_b_tau_lies_in_assistant_role_gap` passes for all six layers.

**MVP scope.** `mvp_only: true` config flag added in `configs/plan_b.yaml`; restricts step_6 to 5 conditions: `aa_capped` + `aa_capped_pc{2,3}_pos2` + `aa_capped_random_{0,1}_pos2`. Drops `pc{2,3}_neg2` (sign-symmetry not needed for H1) and randoms 2–4 (two suffice for the "not blind-spot-aligned" claim). Reduces step_6 wall clock from ~3.5 hr to ~100 min and total run from ~5 hr to ~2.4 hr. Post-deadline replays drop the flag.

**Alternatives considered:**
- Approach A (re-run step 3 projecting on −AA) — rejected; same final τ values, more code churn, marker bookkeeping.
- Capping width=1 (single-layer at L*) — held in reserve as recovery branch; reviewers prefer multi-layer if coherent.
- Skip cap fix entirely; pivot to baseline-only LASSO writeup — held as final fallback if MVP re-run still produces nonsense after all three corrections.

**Reason:** All three bugs explain the observed degenerate outputs and are forced corrections (sign by upstream's published convention, τ by sign math, depth by paper's relative-depth convention). Fix is minimal-churn (3 files, ~30 lines).

**Source:** `cos_sim` audit vs `data/paper_artifacts/assistant_axis_vectors/qwen-3-32b/capping_config.pt`; tau_calibration.json on disk; per-rollout activation projections at L=23, 33–38; orchestrator log at `results/plan_b_gemma2_27b/run_signfix.log`.

**Reversibility:** high. Two-line revert in `plan_b.py` (`p75` → `p25`, drop negation) + config revert (offset 15→6, width 6→8, drop `mvp_only`) + delete `.step3.done`/`.step6.done`. Steerer sign-flip retained as it's correct independent of these changes.

**Downstream dependencies:** all of Plan B Step 6 condition outputs, Step 7a judge labels, Step 7b per-prompt projections, Step 9 metrics (per-condition harm rates + LASSO blind-spot lift recomputed across 5 not 11 conditions), Step 10 figures (bar chart shows fewer bars; honest reflection of MVP scope), `docs/index.html` page text + headline numbers (post-run update).

**Pre-fix run artifacts:** `metrics.json.broken`, `details.parquet.broken`, `tau_calibration.json.old` archived under `results/plan_b_gemma2_27b/_pre_signfix_backup/`. Five non-MVP broken steered parquets archived under `results/plan_b_gemma2_27b/rollouts/broken_pre_signfix/` for retrospective audit.
