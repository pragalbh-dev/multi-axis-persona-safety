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

