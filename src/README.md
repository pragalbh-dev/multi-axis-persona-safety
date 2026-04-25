# `src/` — module map

The architecture this directory locks in is:

```
configs/   (locked YAML; subjects, hooks, eval sizes, capping ranges)
   │
   ▼
src/utils/        ← config validation (pydantic v2), manifest, results-dir contract, env pinning
   │
   ▼
src/data/         ← dataset builders (DAN, Shah-reconstructed)
src/extraction/   ← activation extraction (HF/TL backend + vLLM stub)
   │
   ▼
src/steering/     ← steering / capping (wraps external/assistant-axis ActivationSteering)
   │
   ▼
src/evaluation/   ← phased judge driver, safety harness, capability harness, full driver
   │
   ▼
src/analysis/     ← bootstrap CIs, logistic LASSO, correlations, blind-spot lift
   │
   ▼
src/visualization/ ← matplotlib (paper) + plotly (dashboard) figure rendering
   │
   ▼
src/experiments/  ← exp1..exp6 entry points; configure-and-run, no logic
```

**Import-direction rule.** Modules import from layers above them and from
`utils/`. They do NOT import from peers or layers below: `evaluation` may use
`extraction` and `steering`; `extraction` does not import `evaluation`. This
is enforced by code review, not by tooling — keep it clean.

## Reuse mandate

The vendored upstream `external/assistant-axis/assistant_axis/` covers most
of the activation/steering/PCA work. Stage 1 wraps it; do NOT reimplement.

| What you need | Upstream module | Our wrapper |
| --- | --- | --- |
| Forward-hook activation extraction (HF) | `assistant_axis.internals.activations.ActivationExtractor` | `src.extraction.backend_hf` |
| Steering + capping context manager | `assistant_axis.steering.ActivationSteering` | `src.steering.steerer` |
| Assistant Axis math (compute, project, save) | `assistant_axis.axis` | re-exported from `src.steering` |
| PCA + scalers | `assistant_axis.pca` | used by `src.analysis` |
| vLLM batch generation | `assistant_axis.generation.VLLMGenerator` | called from `src.evaluation.capability` |

## Module responsibilities

### `src/utils/`
- `env.py` — pins `CUDA_VISIBLE_DEVICES=0,1,2,3` at import time + loads `.env`. Every script imports this first.
- `config.py` — pydantic v2 `ExperimentConfig` + `load_experiment_config(path)`. Reads `configs/{subjects,model_hooks,eval_sizes}.yaml` to validate.
- `manifest.py` — `Manifest` dataclass + IO; `is_resumable(dir)`; `current_git_sha()`.
- `results.py` — `init_results_dir(cfg)` enforces the per-experiment output contract (config.yaml + manifest.json + figures/).

### `src/extraction/`
- `types.py` — `ActivationCache` (safetensors + .meta.json round-trip), `ExtractionConfig`.
- `extractor.py` (Stage 1 T1.2 stub) — top-level dispatcher; chooses backend.
- `backend_hf.py` (stub) — wraps `assistant_axis.internals.activations.ActivationExtractor`.
- `backend_vllm.py` (stub) — vLLM hidden-states / nnsight path; Stage 3 T3.1 fills the empirical hook paths.

### `src/steering/`
- `types.py` — `SteeringConfig`.
- `steerer.py` (Stage 1 T1.3 stub) — re-exports `ActivationSteering`; provides `from_config`, `cap_and_steer`, `multi_axis_cap`.

### `src/evaluation/`
- `judge_batch.py` — Stage 0 phased judge driver (`run_judge_batch`); the canonical config-dataclass + phased-load pattern.
- `types.py` — `PER_PROMPT_COLUMNS` (locked), `SafetyResult`, `CapabilityResult`, `EvalResult`.
- `safety.py` (Stage 1 T1.4 stub) — runs DAN + Shah-reconstructed via `run_judge_batch`.
- `capability.py` (Stage 1 T1.4 stub) — adapters for IFEval / MMLU Pro / GSM8k / EQ-Bench keyed off `configs/eval_sizes.yaml`.
- `full.py` (Stage 1 T1.4 stub) — phased driver: subject → safety → capability → cross-check.

### `src/analysis/`
- `types.py` — `BootstrapResult`, `CorrelationResult`, `LassoFit`, `BlindSpotLift`.
- `bootstrap.py` (Stage 1 T1.5 stub) — BCa 10K resample helper.
- `correlation.py` (stub) — Pearson, point-biserial, Kendall τ + BH-FDR.
- `lasso.py` (stub) — nested 10-fold CV logistic LASSO; ordinal LASSO for the secondary check.
- `blind_spot.py` (stub) — `AUC(AA + PCs) − AUC(AA only)` with bootstrap CI.

### `src/visualization/`
- `types.py` — `FigureSpec`, `FigureKind`.
- `figures.py` (Stage 1 T1.5 stub) — `make_figure(spec, data)` returns `(matplotlib_fig, plotly_fig)`.

### `src/data/`
- `build_dan_jailbreak.py`, `reconstruct_shah_jailbreaks.py` — Stage 0 deliverables; do not modify in Stage 1+.

### `src/experiments/`
- `expN_name.py` — entry points. Each is a short script: load config, init results dir, drive `eval_full`, run analysis, render figures, finalize manifest.

## Locked schemas

- **Activation cache:** `data/cache/activations/{model_id}/{dataset}/L{layer}.{safetensors,meta.json}`. Tensor shape `(n_prompts, d_model)` after token aggregation.
- **Experiment config:** `configs/experiment_template.yaml` ↔ `src.utils.config.ExperimentConfig`.
- **Per-prompt result row:** `src.evaluation.types.PER_PROMPT_COLUMNS`.
- **Result-dir contract:** `results/exp{N}_{name}/{config.yaml,manifest.json,metrics.json,details.parquet,figures/}`.
- **Report figure numbering:** `report/figures.md` (Fig 1..6 ↔ Viz 1..6).
