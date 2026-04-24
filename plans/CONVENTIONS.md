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

---

## Decide and log (decide on the go, record here once decided)

These aren't locked — the agent running the relevant stage decides. But once decided, append to this file under the relevant heading so later stages inherit.

### Python version
> _Decided Stage 0: <version>. Reason: <compatibility note>._

### Inference engine
> _Decided Stage 0: vLLM <x.y.z> / SGLang <x.y.z>. Reason: day-0 support for <models>, batch throughput <tokens/sec at util %>._

### Model IDs (exact HuggingFace paths)
> _Decided Stage 0._
> - Tier 1 Gemma 2 27B IT: `<hf-id>`
> - Tier 1 Qwen 3 32B: `<hf-id>` (thinking mode: OFF for reproduction)
> - Tier 1 Llama 3.3 70B IT: `<hf-id>`
> - Tier 2 Gemma 4 31B IT: `<hf-id>` (run both thinking ON + OFF)
> - Tier 2 Qwen 3.6-35B-A3B MoE: `<hf-id>` (run both thinking ON + OFF)
> - Judge primary Qwen 3.6-27B dense: `<hf-id>` (thinking OFF default)
> - Judge cross-check Gemma 4 31B IT: same as Tier 2 subject, separate server instance

### Eval dataset IDs
> _Decided Stage 0._
> - Shah et al. 1,100 persona jailbreaks: `<hf-id or path>`
> - IFEval 541: `<hf-id>`
> - MMLU Pro 1,400 subsample: `<hf-id>`
> - GSM8k 1,000 subsample: `<hf-id>`
> - EQ-Bench 171: `<hf-id>`

### Judge prompt template
> _Decided Stage 2 when safety eval harness implemented. Record: exact prompt, parsing rules, label set (e.g., harmful/refusal/ambiguous), and agreement rule (when to use cross-check judge)._

### Activation cache safetensors schema
> _Decided Stage 2 when extraction harness implemented. Record: tensor dtype (bf16 / fp16 / fp32 tradeoff), shape convention (n_prompts × n_layers × d_model OR one file per layer), metadata fields in sibling `.meta.json`._

### Max input/output lengths per task
> _Decided Stage 0 after token-distribution audit._
> - Persona jailbreak eval: input=<p99 tokens>, output=<tokens>
> - IFEval / MMLU Pro / GSM8k / EQ-Bench: one entry each
> - Judge call: input=<typical>, output=<max, e.g., 128 for label + brief rationale>

### Batch size & TP per model
> _Decided Stage 0 after GPU util baseline test._
> - Each subject and judge model: batch size, tensor_parallel_size, VRAM headroom

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
