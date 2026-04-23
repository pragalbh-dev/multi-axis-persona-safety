# Stage 1: Architecture & Wireframing

**Objective:** Design the codebase structure, evaluation harness, visualization dashboard, report template, and interactive demo data schema. This is the blueprint — no experiment code yet, but the shared infrastructure design is locked.

**Prerequisites:** Stage 0 complete (environment works, models load, codebases cloned).

**Completion criteria:** Every stage 2+ implementer knows: what files to create, what interfaces to implement, what data formats to produce/consume, what the report looks like.

---

## Tasks

- [ ] T1.1: Design codebase structure
  - Define module interfaces for: extraction, steering, evaluation, analysis, visualization
  - Create `src/` subdirectories with `__init__.py` and docstring stubs
  - Define shared data types (e.g., `ActivationCache`, `SteeringConfig`, `EvalResult`)
  - Write a `README.md` for `src/` explaining the module architecture

- [ ] T1.2: Design activation extraction pipeline
  - Interface: `extract_activations(model, prompts, layers) → ActivationCache`
  - Support: single prompt, batch, specific layers or all layers
  - Output format: dict mapping `(prompt_id, layer) → tensor`
  - Save/load format: HDF5 or safetensors for large activation caches
  - Must work with both TransformerLens and nnsight backends

- [ ] T1.3: Design steering mechanism
  - Interface: `steer(model, direction_vector, strength, layers) → steered_model`
  - Support: single PC, multiple PCs, activation capping (min clamp), arbitrary direction
  - The capping formula: `h ← h - v * min(⟨h,v⟩ - τ, 0)` — must be a first-class operation
  - Must be composable: cap PC1 AND steer PC2 simultaneously

- [ ] T1.4: Design evaluation harness
  - **Safety eval interface:** `eval_safety(model, prompts, judge_config) → SafetyResults`
    - Calls judge API in batches, handles rate limiting, retries
    - Returns: per-prompt harm score, aggregate harm rate, bootstrap CI
  - **Capability eval interface:** `eval_capability(model, benchmark, config) → CapabilityResults`
    - Supports: IFEval, MMLU Pro, GSM8k, EQ-Bench
    - Returns: per-benchmark score, aggregate
  - **Combined interface:** `eval_full(model, steering_config) → FullResults`
    - Runs both safety and capability, saves everything to `results/`
  - Results format: JSON lines or parquet, one row per (prompt, condition)
  - Must save full outputs (input, response, PC projections, safety score) for Viz 6

- [ ] T1.5: Design results logging and analysis framework
  - Each experiment writes to `results/exp{N}_{name}/`
  - Standard output files: `config.yaml` (what was run), `metrics.json` (aggregate), `details.parquet` (per-prompt)
  - Analysis module reads results and produces: bootstrap CIs, Cohen's d, correlations, LASSO fits
  - Plotting module produces: matplotlib static figures (for paper) + Plotly interactive (for dashboard)

- [ ] T1.6: Wireframe the report structure
  - Create `report/paper.md` (or `.tex`) with section headings and placeholder text
  - Each section has a "data needed" comment showing which experiment fills it
  - Create `report/blog.md` with a simpler structure
  - Define figure numbering: Fig 1 = Persona Space 3D, Fig 2 = Safety Heatmap, etc.

- [ ] T1.7: Design the interactive demo data schema
  - Define the precomputed output tuple: `(model, steering_mode, strength, defense_config, prompt_id, response_text, pc_projections, safety_score, capability_score)`
  - Storage format: parquet file(s) in `dashboard/data/`
  - Estimate size: ~7,500 entries × ~2KB each ≈ 15MB — easily hostable
  - Wireframe the Dash layout: model dropdown, steering dropdown, strength slider, defense toggle, prompt selector, output panel, PCA mini-plot

---

## Expected Outputs

- `src/` with module stubs and interface docstrings
- `src/README.md` explaining the architecture
- `report/paper.md` and `report/blog.md` with section structure
- `dashboard/` wireframe (layout sketch, data schema)
- `configs/experiment_template.yaml` — template config for any experiment

---

## Notes

- The key design principle: every experiment is just "configure and run." The infrastructure does the heavy lifting (extract → steer → eval → log → analyze → plot). Experiment scripts should be short.
- The evaluation harness must save full (input, response, projections) tuples — not just aggregate metrics — because Viz 6 needs them.
- Consider using Hydra or simple YAML for experiment configs. Don't over-engineer — YAML + argparse is fine.
- The analysis module should produce both static (matplotlib for paper) and interactive (Plotly for dashboard) versions of every figure.
