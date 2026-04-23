# Multi-Axis Persona Safety

Extending the Assistant Axis paper (Lu et al., arXiv 2601.10387) to study the safety-relevant geometry of persona space beyond PC1. Core question: does PC1-only activation capping have blind spots, and can multi-axis capping do better?

**Wiki scope doc:** `~/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`
**Conversation history:** `~/obsidian-vault/raw/conversations/multi-axis-persona-safety-scoping-2026-04-22.md`
**Research ideas basket:** `~/obsidian-vault/wiki/syntheses/research-ideas-basket.md`

> **Note on paths:** `~/obsidian-vault/` is a symlink on both laptop and GPU pointing to the actual obsidian vault (laptop: `~/Documents/knowledge-base/obsidian-vault/`, GPU: `~/data-science/research/knowledge-base/obsidian-vault/`). If the symlink is missing, create it: `ln -s <actual-vault-path> ~/obsidian-vault`.

---

## Current State

**Active stage:** Stage 0 — Environment Setup
**Active task:** Not started
**Last updated:** 2026-04-22

Read `plans/plan.md` for the full stage overview. Read the active stage plan for task details.

---

## Key Decisions (locked — do not change without reading scope doc)

### Models
- **Tier 1 (reproduce):** Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. Thinking mode **OFF** on Qwen 3 for fidelity to paper (§8.1 Limitations: "in Qwen's case, we disabled thinking mode").
- **Tier 2 (extend):** Gemma 4 31B Dense, Qwen 3.6-35B-A3B MoE. Each run with thinking **ON and OFF** — this gives 4 experimental conditions from 2 model weights and cleanly isolates the reasoning axis from model-family confounds.
- **Tier 2 rationale:** paper's §8.1 explicitly names frontier / MoE / reasoning as untested. Tier 2 fills both gaps (MoE + reasoning) with two models.
- **Hardware:** 4x RTX 5090 (32GB each = 128GB total)

### Evaluation
- **Safety:** 1,100 persona-based jailbreak prompts (Shah et al.) — same as the paper
- **Capability:** IFEval (541), MMLU Pro (1,400), GSM8k (1,000), EQ-Bench (171)
- **Judge primary:** self-hosted `Qwen/Qwen3.6-27B` dense, thinking OFF. Benchmarks higher than gpt-4.1-mini (paper's judge), so no methodology regression.
- **Judge cross-check:** self-hosted `google/gemma-4-31b-it`, thinking OFF, on a 200-sample subset per experiment. Inter-judge agreement is reported as a robustness measure — this is a free upgrade over the paper, which used a single judge with no validation.
- **Self-preference handling:** when Gemma 4 31B-it is the *subject* model, skip Gemma-as-judge pass on those prompts (use Qwen primary only). Report which cells had single vs dual judge.
- **Zero external API spend.** Judges run on our GPUs.

### Inference & Serving
- **Engine:** vLLM or SGLang — decided at Stage 0 based on day-0 support for all 5 target models + both judges. Log decision + reason in `plans/CONVENTIONS.md`.
- **Batched inference mandatory.** Single-example inference only in notebooks/tests.
- **Target ≥90% GPU utilization** on steady-state runs. Tune batch size / KV cache / TP before proceeding if under.
- **Sizing:** `max_input_len` and `max_output_len` set per task from a token-distribution audit (Stage 0 T0.2), not guessed. Larger than needed wastes KV-cache memory.
- **Topology:** judge server(s) run persistently on dedicated GPUs; experiment loads co-locate on remaining GPUs. 4× RTX 5090 = 128GB → typically 2 GPUs for judge, 2 for subject, with tensor parallelism per side.

### Tooling Versions & Env
- **Package manager:** `uv` with `pyproject.toml` + lockfile committed.
- **Python version:** pinned at Stage 0 based on vLLM/SGLang/TransformerLens/nnsight compat (likely 3.11 or 3.12).
- **Code style:** `ruff` (format + lint), `mypy` strict on `src/`, `pytest` for tests.
- **Secrets:** `.env` (gitignored) + `.env.example` (committed). Required: HF token only. No external LLM API keys.
- **Seeds:** every experiment config has a `seed` field; set torch + numpy + python random + `PYTHONHASHSEED` + engine seed; logged in `manifest.json`.

### Data & Checkpointing
- **Cached activations:** parquet in `data/cache/activations/`, one file per (model, dataset, layer).
- **Results layout:** every `results/exp{N}_{name}/` contains `config.yaml`, `manifest.json` (schema + seed + git SHA + artifacts), `metrics.json`, `details.parquet`, `figures/`.
- **Checkpointing principle:** any unit of work >15 min to redo gets a checkpoint; nothing smaller. Checkpoint after (model, layer) extraction cells, after each steering-strength × layer cell, after full 1,100-prompt judge passes. Do not dump per-prompt intermediates during a run.
- **Resume:** experiment scripts check for existing `manifest.json` and resume from last checkpoint unless `--fresh` is passed.

### Statistical Framework
- Marchenko-Pastur threshold for dimensionality
- BH-FDR correction (q=0.05) for multiple PC testing
- Bootstrap BCa CIs (10K resamples)
- Cohen's d effect sizes
- Random vector baselines (matched norm)
- LASSO with nested 10-fold CV for joint prediction

### Hyperparameters (calibrated, not fixed)
- Activation cap threshold (tau): sweep 1st/10th/25th/50th/75th percentile per PC independently
- Steering strength (lambda): -2 to +2 in 0.5 steps
- Layer selection: sweep 50-90% depth range

### Tooling
- **Activation extraction:** TransformerLens (primary), nnsight (backup/MoE)
- **Starting codebases:** github.com/safety-research/assistant-axis, github.com/safety-research/persona_vectors
- **Visualization:** Plotly Dash
- **Interactive demo:** Precomputed outputs, hosted on HuggingFace Spaces

### Threat Model
- Activation-level attacks on open-weight models only
- Per Non-Surjective paper (2604.09839): steered states have no prompt pre-image — do NOT claim findings transfer to prompt-only access

---

## Code Conventions

### Directory structure
```
src/
  extraction/     ← Activation extraction pipeline
  steering/       ← Steering mechanism (add λ·v to residual stream)
  evaluation/     ← Safety + capability eval harness
  analysis/       ← Statistical analysis (bootstrap, LASSO, PCA)
  visualization/  ← Dashboard components
  utils/          ← Shared utilities
configs/          ← Model configs, eval configs, hyperparams (YAML)
data/             ← Cached activations, downloaded eval datasets (gitignored)
results/          ← Experiment outputs, per-experiment subdirs (gitignored)
report/           ← Paper draft, blog draft
dashboard/        ← Plotly Dash app
notebooks/        ← Exploratory analysis (not production code)
plans/            ← Project plans and progress ledger
```

### Naming
- Python files: `snake_case.py`
- Config files: `snake_case.yaml`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Experiment result directories: `results/exp{N}_{name}/` (e.g., `results/exp1_pca_decomposition/`)

### Running experiments
- Each experiment has a main script: `src/experiments/exp{N}_{name}.py`
- Config for each experiment: `configs/exp{N}.yaml`
- Results saved to: `results/exp{N}_{name}/`
- All experiments use the shared eval harness in `src/evaluation/`

### Progress tracking
After completing any task:
1. Check the box in the relevant `plans/stage-{N}-*.md`
2. Update "Current State" in this file (active stage, active task, date)
3. Append a line to `plans/progress.md`

---

## Do NOT

- Change the eval benchmarks or statistical framework without reading the scope doc
- Restructure `src/` without updating the stage plans
- Skip random baselines in any steering experiment
- Claim prompt-level attack implications from steering results
- Commit large files to git (activations, model weights go in data/ which is gitignored)
- Amend previous commits — always create new ones
