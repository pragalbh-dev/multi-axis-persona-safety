# Multi-Axis Persona Safety

Extending the Assistant Axis paper (Lu et al., arXiv 2601.10387) to study the safety-relevant geometry of persona space beyond PC1. Core question: does PC1-only activation capping have blind spots, and can multi-axis capping do better?

**Wiki scope doc:** `~/Documents/knowledge-base/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`
**Conversation history:** `~/Documents/knowledge-base/obsidian-vault/raw/conversations/multi-axis-persona-safety-scoping-2026-04-22.md`
**Research ideas basket:** `~/Documents/knowledge-base/obsidian-vault/wiki/syntheses/research-ideas-basket.md`

---

## Current State

**Active stage:** Stage 0 — Environment Setup
**Active task:** Not started
**Last updated:** 2026-04-22

Read `plans/plan.md` for the full stage overview. Read the active stage plan for task details.

---

## Key Decisions (locked — do not change without reading scope doc)

### Models
- **Tier 1 (reproduce):** Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B
- **Tier 2 (extend):** Gemma 4 31B Dense, Qwen 3.6-35B-A3B MoE
- **Hardware:** 4x RTX 5090 (32GB each = 128GB total)

### Evaluation
- **Safety:** 1,100 persona-based jailbreak prompts (Shah et al.) — same as the paper
- **Capability:** IFEval (541), MMLU Pro (1,400), GSM8k (1,000), EQ-Bench (171)
- **Judge:** GPT-4.1-mini or Gemini 2.5 Flash primary; frontier model on 200-sample validation subset

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
