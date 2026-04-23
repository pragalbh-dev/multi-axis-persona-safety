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
- **Cached activations:** parquet in `data/cache/activations/`. One file per (model, dataset, layer) triple. Schema logged to `CONVENTIONS.md` when first written (see "Decide and log" below).
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

### Activation cache parquet schema
> _Decided Stage 2 when extraction harness implemented. Record: columns, dtypes, compression, chunking._

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
