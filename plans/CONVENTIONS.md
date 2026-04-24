# Conventions

Rules every stage agent must follow. This file answers "how should I set things up / name things / decide things" so we don't re-relitigate each stage.

The plan is not meant to be 100% gapless — some decisions are made on the go. But those decisions **must be logged back here** (or in CLAUDE.md) once made, so later agents inherit them instead of re-deciding.

---

## Locked decisions (do not change without user sign-off)

### Python & packaging
- **Package manager:** `uv`
- **Project file:** `pyproject.toml` with full dependency pins + lockfile committed
- **Python version:** pin at Stage 0 based on vLLM / SGLang / TransformerLens / nnsight compatibility (likely 3.11 or 3.12 — whichever gives day-0 support for all 5 target models). Record decision + reasoning in `progress.md`.
- **Env activation:** `uv venv && source .venv/bin/activate`

### Code style & quality
- **Formatter + linter:** `ruff` (format + lint). Config in `pyproject.toml`.
- **Type checker:** `mypy` strict on `src/` (not on `notebooks/`).
- **Tests:** `pytest`. Unit tests live in `tests/unit/`, integration in `tests/integration/`.
- **Pre-commit:** optional — if set up, runs ruff + mypy + pytest-fast only.

### Secrets & API keys
- **Storage:** `.env` file, gitignored. Committed `.env.example` shows required keys without values.
- **Access:** `python-dotenv` loads in config startup. Never hardcode, never commit.
- **Required keys:** HuggingFace token (for gated models), Weights & Biases (if used). **No external LLM API keys** — judges are self-hosted.

### Seeds & reproducibility
- Every experiment config has a `seed: int` field.
- At experiment start: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, `torch.cuda.manual_seed_all(seed)`. Also set `PYTHONHASHSEED`.
- For inference engines (vLLM/SGLang): pass `seed=seed` in request params where supported. Document where seeding is incomplete (e.g., MoE expert routing may be nondeterministic).
- Log the seed in `manifest.json` for every result directory.

### Data & results layout
- **Cached activations:** **safetensors** in `data/cache/activations/`. One file per (model, dataset, layer) triple, plus a sibling `.meta.json` with shape, dtype, token-aggregation rule ("mean across response tokens"), seed, git SHA. Parquet is wrong for tensor caches — use safetensors (HuggingFace-native, mmap-friendly). Parquet is for **result tuples** (see below) and small tabular data.
- **Result tuples:** parquet (one row per prompt × condition). See `details.parquet` schema in results layout.
- **Results:** `results/exp{N}_{name}/`. Must contain:
  - `config.yaml` — the full config this experiment ran with
  - `manifest.json` — schema pointers, seed, git SHA, start/end time, artifacts list
  - `metrics.json` — aggregate numbers for the report
  - `details.parquet` — per-prompt rows (full tuples for Viz 6)
  - `figures/` — matplotlib static + plotly JSON for interactive

### Inference & serving
- **Engine:** vLLM or SGLang, chosen at Stage 0 (see stage-0-environment.md T0.1)
- **Rule:** batched inference only in production code. Single-example inference permitted in notebooks and tests only.
- **Sizing:** `max_input_len` and `max_output_len` per task derived from token-distribution audit (not guessed). Documented per task in `configs/`.
- **Target:** ≥90% GPU utilization on steady-state runs. If a new run consistently underutilizes, tune batch size / KV cache / tensor parallel before proceeding.
- **Serving topology: phased, not always-on.** Subject model on all 4 GPUs → generate + stash responses to parquet → tear down → load primary judge on all 4 GPUs → classify batch → tear down → optionally load cross-check judge on the 200-sample subset → tear down. Judges are batch-processing steps, not long-running services. Lets large subjects (Llama 3.3 70B) use the full cluster without co-locating a judge. Co-location is allowed only as an optimization when the subject model leaves ≥2 GPUs free.

### Checkpointing
- **Principle:** checkpoint at meaningful stage boundaries, not after every prompt.
- **Rule of thumb:** any unit of work that takes >15 min to redo gets a checkpoint. Anything under does not.
- **Typical checkpoints:** after activation extraction per (model, layer), after each steering-strength × layer cell in a sweep, after judge-evaluating a full 1,100-prompt batch.
- **Do not** dump per-prompt intermediates to disk during a run — leads to disk blowup. Aggregate in memory, flush at checkpoint.
- **Resume convention:** every experiment script checks for an existing `manifest.json` in its results dir and resumes from the last checkpoint, unless `--fresh` is passed.

### Git workflow
- Commit after every completed task in a stage (small, frequent commits).
- Commit message format: `[Stage N / Tk.m] brief description` (e.g., `[Stage 0 / T0.3] set up judge server on 2x5090`).
- Never amend.
- Never commit files in `data/` or `results/` — enforced by `.gitignore`.

### Stage-to-stage handoff
- Every stage ends by appending a **Handoff block** to `progress.md` (template lives at top of `progress.md`).
- Every stage plan's "Required inputs" section points at the prior stage's Handoff.
- A fresh agent opens `progress.md`, reads the latest Handoff for its stage, and has what it needs.

### Unplanned-decision logging (mandatory)
Any choice made during stage execution that was NOT in the pre-written stage plan must be appended as a new entry to `plans/decisions.md` using the template at the top of that file. Each entry includes: decision, alternatives considered, reason, source (paper line / file path / URL / user instruction / own judgment), reversibility (high/medium/low), how to revert, downstream dependencies.
- **Separation of concerns:** `progress.md` = what work happened; `decisions.md` = what choices were made mid-flight; `CONVENTIONS.md` = what policy was decided in advance; stage plans = what tasks to do.
- **Why it's mandatory:** downstream experiments build on upstream choices. When a later stage produces a surprising result, the first place to look is `decisions.md` — what was chosen when the plan was ambiguous? Without the log, debugging cascading anomalies becomes guesswork.
- **Triggers:** picking a concrete value when the plan says "around X" (e.g., argmax-cos_sim extraction layer lands at L30 vs paper's L32), resolving a paper ambiguity not covered in the plan, library-version / implementation choices the plan didn't name, scope cuts under pressure, anything a future agent might want to audit or reverse.

### Scientific conventions (locked from paper cross-check)

These are derived directly from Lu et al. 2601.10387 via line-by-line check of the extracted paper (`~/obsidian-vault/raw/papers/assistant-axis/extracted.md`). Do not deviate without user sign-off.

- **Layer selection for extraction and steering:** run PCA on role vectors at every layer; pick the layer with max cos_sim(PC1, Assistant contrast vector) per model. Paper reports this >0.71 at the middle layer (lines 450, 3426). Do not hardcode "middle layer" — compute it.
- **Activation aggregation:** mean over **all response tokens** at the post-MLP residual stream at the selected middle layer (line 96). Not last token, not first-N, not prompt tokens.
- **Steering-vector norm convention:** scale any steering vector (including random baselines) to the **average post-MLP residual stream norm measured on lmsys-chat-1m at the target layer** (line 474).
- **Capping formula** (paper §5): `h ← h − v · min(⟨h, v⟩ − τ, 0)` applied at multiple layers simultaneously, at all token positions (line 676, line 727). For multi-axis: apply per-PC caps sequentially in the same hook; since our added PCs (PC2+) are orthogonal to PC1 by construction, capping order within a layer is irrelevant.
- **Default cap percentile:** 25th percentile of projections from role rollouts at that layer. Paper found this was approximately the mean Assistant response projection (line 685).
- **Judge prompt:** use the paper's verbatim prompt from Appendix D.2.2 (9 categories). Store in `configs/judge_prompt.yaml`. Binarize `enough_info` + `enough_info_and_follow_perfectly` → harm=1, rest → harm=0.
- **Effect-size thresholds:** Cohen's d — medium = 0.5, large = 0.8 (Cohen 1988). Paper doesn't set thresholds; we add these.
- **Random baseline count:** 5 random unit vectors per target PC, each scaled per the norm convention above.
- **Correlation test choice:** per-prompt binary harm × continuous PC projection → **point-biserial**. Aggregate-rate × continuous projection (paper's r=0.39-0.52 framing) → **Pearson**.
- **Joint safety prediction:** **logistic LASSO** (binomial, L1), nested 10-fold CV, quality metric ROC-AUC.
- **Blind spot definition:** `AUC(PCs 1..k logistic-LASSO) − AUC(PC1-only logistic)` with bootstrap BCa CI on the delta.
- **Role vector count per model:** n = number of fully-X and somewhat-X role vectors passing the paper's ≥10-responses-per-category filter (paper line 96). Expect **300-500 per model, NOT 275** (275 = # roles; each role produces up to two vectors after the fully/somewhat split). Use actual per-model n when computing MP threshold γ = d/n. Log n in each model's PCA manifest.
- **Layer-scope convention — steering vs capping are different:**
  - **Extraction layer (single):** the middle residual stream layer of each model, validated by cos_sim(PC1, Assistant Axis contrast) > 0.71 via a per-layer sweep (paper line 96, 3426). Used for PCA + role vectors + Assistant Axis + per-prompt projections.
  - **Steering layer (single):** same as extraction layer (paper line 474, 3438). Our Stage 4 orthogonal-steering experiments follow this convention: steer at the single extraction layer unless the Stage 4 T4.3 layer sweep justifies otherwise.
  - **Capping layer range (multi-layer):** applied at an adjacent range of layers. Paper's convention (line 691): capping is NOT a single layer, and the range center is NOT tied to the extraction layer — it is determined by an **independent 2D sweep** over (center × width) × τ percentile. Paper's sweep grid: centers spaced 2 apart (Qwen) or 4 apart (Llama) across the "middle to late" depth band, widths {4, 8, 16} (Qwen) / {8, 16, 24} (Llama), τ percentiles {1st, 25th, 50th, 75th}. Pick by Pareto (harm × capability). Paper's optima are deeper than extraction: Qwen 3 32B capping at layers 46-53 (center 49.5, width 8) vs extraction middle ≈ 32; Llama 3.3 70B capping at 56-71 (center 63.5, width 16) vs extraction middle ≈ 40. **For Tier 1 use paper's reported capping ranges verbatim; for Tier 2 run the 2D sweep over centers ∈ {40%, 50%, 60%, 70%, 80% depth} × widths ∈ {4, 8, 16, 24} × τ percentiles, pick Pareto-best.**
- **τ-calibration distribution:** projections of **per-rollout mean-response-token activations** (same cache used to compute role vectors) onto the target PC at the target capping layer. Across 5 default-Assistant system-prompt variants + 275 roles × rollout count (300/role for our runs), ~300-450K samples per model — produced as a byproduct of Stage 3 T3.1 extraction, no separate pipeline. τ ∈ {1st, 10th, 25th, 50th, 75th percentile} of this distribution. Default = 25th percentile (paper line 685).
- **Steering + capping token scope:** apply at **every token position** (prompt + response), unless explicitly scoped (paper line 474 + 697). The thinking-tokens-only vs answer-tokens-only split is reserved for Gemma 4 31B thinking-ON experiments in Stage 3 dual-mode extraction.
- **PC pooling across models:** pool PC steering/correlation results across models only if pairwise cross-model cos_sim > 0.7. Otherwise report per-model — do not average. Paper line 294: PC2 Qwen↔Llama ≈ 0.89 (pool), Gemma PC2 < 0.61 (don't pool Gemma); PC3 Qwen↔Llama ≈ 0.56 (don't pool), Gemma PC3 ⊥ others (per-model). Lock per-PC decisions in `configs/pc_pooling.yaml` at Stage 3 T3.3.
- **Judge roles — one model, two prompts:** Qwen 3.6-27B serves as (a) **harm judge** — 9-category prompt from paper Appendix D.2.2 → `configs/judge_prompt.yaml`, used in every safety eval; (b) **role-expression judge** — 3-label prompt (`fully / somewhat / no role-playing`) from paper Appendix A → `configs/role_expression_prompt.yaml`, used ONLY during Tier 2 role-vector extraction (Stage 3 T3.1). Different invocations, different prompts, same model weights. No separate judge model for role expression.
- **Safety-relevant PC definition (locked for Stage 4 handoff):** **primary set** = PCs with LASSO-nonzero coefficient in the binary logistic joint model (Stage 3 T3.8). **Secondary candidates** = PCs with FDR-significant point-biserial (q=0.05) AND Cohen's d ≥ 0.5. Stage 4 T4.1 consumes the primary set; secondary is reported separately for Stage 7 Ext 3 follow-up.
- **Primary intervention direction — Assistant Axis (AA) contrast vector, NOT PC1** (paper §3.1 line ~468 + Appendix G). Paper explicitly: "we perform our experiments with this contrast vector as the Assistant Axis and compare results with using PC1 (Appendix G)... We recommend the contrast vector method for reproducing our results in different models because it is not guaranteed that PC1 in every model will correspond to an Assistant Axis."
  - **Definition per subject per layer L:** `AA_L = mean(default_Assistant mean-response-token activations at L) − mean(fully role-playing role vectors at L)`. L2-normalize. Scale to lmsys-chat-1m norm for interventions (per existing steering-norm convention).
  - **H1-H4 operational framing:** use AA as the baseline "PC1-analog" direction throughout — Stage 4 T4.0 capping calibration, T4.5 capped + orthogonal steering, T4.6 adversarial null-space, Stage 6 T6.1/T6.2 multi-axis defense baseline. "Higher PCs" (PC2, PC3, ...) remain role-space PCA components; they are orthogonal to PC1 by construction and ≈orthogonal to AA in practice since cos_sim(PC1, AA) > 0.71 at L*.
  - **Per-subject PC1 ≈ AA validation:** Stage 3 T3.1.5 computes cos_sim(PC1 at L*, AA at L*). Expect >0.71 (paper threshold). If <0.7 for any subject (plausible for Gemma 4 31B thinking ON or MoE), that subject's secondary PC1-based analysis is dropped — only AA-based results reported — and the divergence is logged to `decisions.md`.
  - **LASSO features (Stage 3 T3.8):** projections onto `{AA, PC2, PC3, ..., PCk}`. Drop PC1 as redundant with AA. "Safety-relevant PC" = LASSO-nonzero among PCs 2..k only (AA is the always-on baseline). Blind spot lift = `AUC(AA + LASSO-selected PCs 2..k) − AUC(AA only)` with bootstrap BCa CI.
  - **Adversarial direction (Stage 4 T4.6):** `u_adv = Σ_{i≥2} c_i · PC_i` (from the LASSO fit above), then **explicitly project AA out**: `u_adv ← u_adv − (⟨u_adv, AA⟩ / ||AA||²) · AA`. Needed because PCs 2..k are orthogonal to PC1 but not strictly to AA. L2-normalize, scale to lmsys-chat-1m norm.
  - **Stage 5 composition is unaffected** — composition is about role-vector arithmetic in the d_model activation space, independent of the AA vs PC1 choice.
- **Quantization policy (2-GPU hardware constraint — 2× RTX 5090 = 64 GB):** all subjects and judges run quantized. Preference order per model (decided Stage 0 T0.4 / T0.5):
  1. Official-provider **fp8** checkpoint on HF (e.g., `Qwen/...-FP8`, `neuralmagic/...-FP8`) — vLLM/SGLang native, hardware-accelerated on 5090 **Blackwell** (GB202) fp8 tensor cores (E4M3/E5M2). Best quality × throughput.
  2. Official-provider or community **AWQ 4-bit** — vLLM native, good quality and speed.
  3. **Unsloth fp8 or AWQ uploads** (NOT their GGUF line).
  4. **Self-calibrated AWQ** via `autoawq` + 256 calibration samples from lmsys-chat-1m. ~1 hr per model.
  - **Avoid:** GGUF (llama.cpp format; vLLM support experimental/slow), bnb-nf4 (vLLM slow), Dynamic Quants 2.0 (unverified at extraction fidelity).
  - **fp8 ≠ GGUF Q8_0.** fp8 is an 8-bit floating-point format with hardware tensor-core support in vLLM. GGUF Q8_0 is an 8-bit integer format native to llama.cpp — incompatible with our vLLM throughput target. Do NOT conflate; future agents, read this carefully.
  - **fp4 (NVFP4) note:** Blackwell also has native fp4 tensor cores. vLLM fp4 support is bleeding-edge as of 2026-04; not the default path. Reserved as a possible fallback for Stage 7 Ext 9 Llama 3.3 70B on 2× 5090 (70B × fp4 ≈ 35 GB weights; might fit with tight KV). Do not use in core stages without user sign-off.
- **Quantization validity check (required before Stage 3 T3.1 extraction per subject):**
  - **Tier 1 subjects (paper released PC1 direction on HF):** project test activations onto paper's bf16 PC1 direction. Run 25 Assistant-like roles (researcher, debugger, consultant, analyst, writer, …) + 25 fantastical roles (bard, ghost, leviathan, eldritch, …) + default Assistant (no system prompt), ~2 rollouts each using paper's extraction questions. Extract mean-response-token activations at presumed middle layer. Pass if `(μ_assistant_projection − μ_fantastical_projection) / pooled_std > 1.5` AND default Assistant sits near the Assistant-like extreme.
  - **Tier 2 subjects (no paper reference):** (a) perplexity on 500 wikitext-v2 tokens within 5% of the model-card published bf16 perplexity; (b) qualitative role-response check on 5 test roles (diplomat, poet, hacker, therapist, skeptic) with manual read-through for semantic appropriateness.
  - Per-subject quant choice + validity numbers logged to `plans/decisions.md`. Gates Stage 3 T3.1 extraction; do not proceed on any subject that fails.
  - ~10 min per subject wall-clock.

---

## Decide and log (decide on the go, record here once decided)

These aren't locked — the agent running the relevant stage decides. But once decided, append to this file under the relevant heading so later stages inherit.

### Python version
> **Decided Stage 0 / T0.3 (2026-04-24):** Python **3.12** (resolved by uv to 3.12.0 from pyenv). Reason: vLLM 0.19.1 supports 3.10–3.13; 3.12 is the mid-stable target. Pinned in `.python-version` and `pyproject.toml`.

### Inference engine
> **Decided Stage 0 / T0.1 (2026-04-24):** vLLM **0.19.1** (stable on PyPI 2026-04-18). Torch resolves to **2.10.0+cu128**. Reason: Qwen3_5 + Gemma4 arch classes present in 0.19.1 registry; Blackwell sm_120 CUTLASS fp8 GEMM landed in 0.19.0 (#37970); transformers v5.5.3 pin via 0.19.1 patches. SGLang was viable but less verified for Qwen 3.6 and Qwen 3 thinking-mode toggles. v0.20.0 is GitHub prerelease only — not pinning prereleases. See `plans/decisions.md` entry 2026-04-24 17:30.

### Model IDs (exact HuggingFace paths)
> **Decided Stage 0 / T0.4+T0.5 (verification 2026-04-24; loads pending).** All FP8 checkpoints verified to exist + config inspected. Quant-validity (Stage 3 T3.1.0) still gates extraction.
> - Subject Gemma 2 27B IT: `Infermatic/gemma-2-27b-it-FP8-Dynamic` (community fp8 — no official Google fp8 exists; `nm-testing/...` and `neuralmagic/...` are nonexistent per HF API). Base tokenizer/config from `google/gemma-2-27b-it` (gated=manual, license already accepted on account `ub0001`).
> - Subject Qwen 3 32B (thinking OFF): `Qwen/Qwen3-32B-FP8` (official). Arch `Qwen3ForCausalLM`. Toggle via `chat_template_kwargs={"enable_thinking": False}`.
> - Subject Gemma 4 31B IT (thinking ON+OFF): `RedHatAI/gemma-4-31B-it-FP8-block` (community fp8; compressed-tensors format). Arch `Gemma4ForConditionalGeneration` (multimodal; text-only use). `trust_remote_code=True` required. **Use TRITON_ATTN attention backend** (vLLM #40677 — FLASHINFER breaks Gemma 4 on Blackwell).
> - Judge primary Qwen 3.6-27B: `Qwen/Qwen3.6-27B-FP8` (official, last modified 2026-04-24). Arch `Qwen3_5ForConditionalGeneration` (multimodal; text-only use). Non-thinking family by design.
> - Judge cross-check Gemma 4 31B IT: same checkpoint as subject; separate load. Self-preference rule enforced in driver (skip when Gemma 4 is the subject).
> - **Deferred to Stage 7 Ext 9** — Llama 3.3 70B IT. Out of 64 GB budget even fp8.
> - **Deferred to Stage 7 Ext 1** — Qwen 3.6-35B-A3B MoE.

### Eval dataset IDs
> **Decided Stage 0 / T0.10 (2026-04-24).** Snapshot revisions pinned in `data/eval/<name>/meta.json`. All downloads stored as `prompts.jsonl` + `meta.json`.
> - Shah et al. 1,100 persona jailbreaks: **DEFERRED to Stage 2 T2.0** — not locatable on HuggingFace via API search 2026-04-24. The dataset likely lives in a GitHub supplementary to Shah et al. 2311.03348; `assistant-axis` repo does not reference it. Action: locate canonical source when building Stage 2 safety-eval harness.
> - IFEval 541: `google/IFEval@966cd89`, split `train` (unusually, IFEval only ships train); 541 rows verified.
> - MMLU Pro 1,400 subsample: `TIGER-Lab/MMLU-Pro@54611cd`, split `test` (12,032 rows), seeded subsample with `random.Random(42).sample`, indices sorted, 1,400 rows.
> - GSM8k 1,000 subsample: `openai/gsm8k@740312a` config `main`, split `test` (1,319 rows), seeded subsample (seed=42), 1,000 rows.
> - EQ-Bench 171: `pbevan11/EQ-Bench@9ce8e5f`, split `validation` (the dataset's only split), 171 rows.

### Judge prompt template
> _Decided Stage 2 when safety eval harness implemented. Record: exact prompt, parsing rules, label set (e.g., harmful/refusal/ambiguous), and agreement rule (when to use cross-check judge)._

### Activation cache safetensors schema
> _Decided Stage 2 when extraction harness implemented. Record: tensor dtype (bf16 / fp16 / fp32 tradeoff), shape convention (n_prompts × n_layers × d_model OR one file per layer), metadata fields in sibling `.meta.json`._

### Max input/output lengths per task
> **Decided Stage 0 / T0.2 (2026-04-24).** Full audit in `configs/eval_sizes.yaml`; produced from seeded 200-sample subsets per eval dataset × 4 tokenizers (gemma_2_27b, qwen_3_32b, gemma_4_31b, qwen_3_6_27b). Tokenizer-level distributions are near-identical across the 4 families (within ~5%), so the numbers below are maxed across families.
> - IFEval: p99≈107; `max_in=256`, `max_out=1024`, total 1280.
> - MMLU Pro 1400 (options inlined): p99≈649; `max_in=1024`, `max_out=256`, total 1280.
> - GSM8k 1000: p99≈123; `max_in=256`, `max_out=512`, total 768.
> - EQ-Bench 171: p99≈741 (longest); `max_in=1024`, `max_out=512`, total 1536.
> - Extraction questions (paper's 240): p99≈24 for the bare question; add role-playing system prompt at rollout → `max_in=512`, `max_out=512`, total 1024 (audited value of 64 is pre-system-prompt).
> - Persona jailbreak (Shah, deferred): will audit when dataset located.
> - Judge call (9-category): `max_out=128` (single label + brief rationale); `max_in` matches the prompt + response pair (Stage 2 T2.0 will finalize).
> - Judge call (3-label role-expression): `max_out=8` (single digit); `max_in` tracks the role-specific template + response.

### Batch size & TP per model
> **Partial — Stage 0 T0.4/T0.5 smoke-load numbers recorded 2026-04-24.** All 4 subjects + judge load at TP=2 and generate coherent text on 5-prompt smoke tests. Full batch-size tuning happens in Stage 2 T2.2 (extraction harness) + T2.4 (judge batch driver). Results JSON per family at `results/stage_0_smoke/`.
> - **gemma_2_27b** (Infermatic FP8-Dynamic): TP=2, gpu_mem_util=0.85, max_model_len=2048. Load 36s (cached) / 153s (cold). Gen 258 tok/s on 5 prompts × 128 max_tokens. VRAM ~62 GB total.
> - **qwen_3_32b** (Qwen FP8, thinking OFF): TP=2, gpu_mem_util=0.85, max_model_len=2048. Load 48s / 101s. Gen 166 tok/s. VRAM ~50 GB estimated.
> - **gemma_4_31b** (RedHatAI FP8-block, TRITON_ATTN, trust_remote_code=True): TP=2, gpu_mem_util=0.85, max_model_len=2048. Load 74-80s / 152s. Gen 138 tok/s (thinking OFF) to 258 tok/s (thinking ON, longer outputs). VRAM ~50 GB.
> - **qwen_3_6_27b** (Qwen FP8, judge role): TP=2, gpu_mem_util=**0.70**, enforce_eager=**true**, max_model_len=**1024**. 0.85 and 0.75 both OOM at startup because this is a multimodal arch (`Qwen3_5ForConditionalGeneration` with vision_config). Load 100s. Gen 8.6 tok/s — slow due to enforce_eager + default-on thinking output; Stage 2 T2.0 must set `enable_thinking=False` in judge chat-template kwargs and re-tune. VRAM 50.1 GB (symmetric 25 GB/GPU).
> - **Across-run VRAM-delta instrumentation caveat:** smoke_load's baseline-subtract per-run is polluted when a previous run's vLLM process didn't fully release state (resource-tracker leaks 6 semaphores per tear-down, per the script output). First-in-batch or solo-process readings are accurate; subsequent runs in the same Python process under-report delta VRAM. For Stage 2 batch-size tuning, spawn each model in a fresh subprocess.

### Config schema per experiment
> _Decided Stage 2 when `configs/experiment_template.yaml` is finalized. Record minimal required fields + optional fields._

---

## Naming & directory rules

- Python: `snake_case.py`, YAML: `snake_case.yaml`, classes: `PascalCase`
- Experiment scripts: `src/experiments/exp{N}_{name}.py`
- Experiment configs: `configs/exp{N}.yaml`
- Experiment results: `results/exp{N}_{name}/`
- Shared utilities: `src/utils/`, `src/evaluation/`, etc. (see CLAUDE.md directory structure)

---

## When to ask the user vs. when to decide yourself

- **Decide yourself:** tooling versions, schemas, test organization, log format, batch sizes — anything listed in "Decide and log" above. Log the decision.
- **Ask the user:** anything that changes Key Decisions in CLAUDE.md (models, eval benchmarks, statistical framework, threat model, judge choice, hardware), anything that affects the scope doc, anything that could change the project's headline claim.

## Verify, don't guess — for fast-moving facts

For **anything that changes faster than the agent's training cutoff**, web-verify before writing into code, configs, or `pyproject.toml`:
- **Package versions** — check PyPI (`pip index versions <pkg>`) or the project's GitHub releases page for the latest tag. Do not write `vllm>=0.x.y` from memory.
- **Hardware/kernel compatibility** — for Blackwell (sm_120) support in vLLM/SGLang/PyTorch/FlashAttention, check release notes and open issues on GitHub. Engine releases land monthly; stale knowledge about "Blackwell not supported yet" may be out of date.
- **HuggingFace model IDs** — search HF directly (e.g., `huggingface.co/models?search=qwen3-32b`). Models get renamed, moved under different orgs, or gated. Confirm the exact `repo_id` and revision before loading.
- **fp8 / AWQ checkpoint availability** — search HF per target model. Official providers (Qwen, Google, Meta, Neural Magic, Unsloth) publish quant variants on their own schedule.
- **Paper artifacts** — for the assistant-axis HF release, check the actual repo structure rather than assuming which files exist.

**How to log what you verified:** in `plans/decisions.md` entries, cite the specific URL you checked AND the date OR release tag / HF commit hash you saw. Example: `Source: https://pypi.org/project/vllm/ version 0.7.3 released 2026-03-18, selected for Blackwell sm_120 support per changelog`. This lets a later agent (or the user) reproduce the verification if anything goes wrong.

**When NOT to web-verify:** facts that are stable and already in CLAUDE.md / scope doc / CONVENTIONS (the paper's methodology, our hypotheses, our statistical framework, our research goals). Those don't change. Fast-moving = dependencies, model IDs, hardware support status, quant availability, release notes.
