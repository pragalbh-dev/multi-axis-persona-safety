# Stage 2: Core Infrastructure Implementation

**Objective:** Build and test all shared components that every experiment needs. After this stage, running an experiment is just "write a config + a short script."

**Prerequisites:** Stage 1 complete (architecture designed, interfaces defined).

**Completion criteria:** End-to-end smoke test passes: load model → extract activations → compute PCA → steer on PC1 → evaluate safety (100 prompts) → evaluate capability (50 problems) → save results → generate one plot. All on Gemma 2 27B with a small sample.

---

## Tasks

- [ ] T2.1: Implement activation extraction pipeline
  - `src/extraction/extractor.py` — TransformerLens backend
  - `src/extraction/extractor_nnsight.py` — nnsight backend (for MoE later)
  - `src/extraction/cache.py` — save/load activation caches
  - Support: batch extraction, layer selection, mean-across-tokens aggregation
  - Test: extract activations for 10 prompts on Gemma 2 27B, verify shape and values

- [ ] T2.2: Implement PCA and projection module
  - `src/analysis/pca.py` — centered PCA, eigenspectrum, Marchenko-Pastur threshold
  - `src/analysis/projections.py` — project activations onto PCs, compute cosine similarities
  - Test: load pre-computed role vectors from assistant-axis HuggingFace, run PCA, verify PC1 matches their reported cosine similarity with the assistant axis (>0.71)

- [ ] T2.3: Implement steering mechanism
  - `src/steering/steerer.py` — add λ·v to residual stream at specified layers
  - `src/steering/capper.py` — activation capping: `h ← h - v * min(⟨h,v⟩ - τ, 0)`
  - `src/steering/multi_axis.py` — compose multiple caps/steers simultaneously
  - Support: TransformerLens hooks, configurable layers, configurable strength
  - Test: steer Gemma 2 27B on PC1 at strength 1.0, verify output changes qualitatively

- [ ] T2.4: Implement safety evaluation harness
  - `src/evaluation/safety_eval.py` — run jailbreak prompts, collect responses
  - `src/evaluation/judge.py` — call judge API (GPT-4.1-mini), parse scores, handle batching/retries/rate-limits
  - `src/evaluation/safety_metrics.py` — aggregate harm rate, bootstrap CI, Cohen's d
  - Test: evaluate 20 jailbreak prompts on Gemma 2 27B (unsteered), verify judge returns parseable scores

- [ ] T2.5: Implement capability evaluation harness
  - `src/evaluation/capability_eval.py` — run benchmark prompts, score automatically
  - Support: IFEval (rule-based scoring), MMLU Pro (multiple choice), GSM8k (numeric answer extraction), EQ-Bench (scoring protocol)
  - Test: evaluate 10 problems from each benchmark on Gemma 2 27B

- [ ] T2.6: Implement results logging
  - `src/utils/logger.py` — write experiment config, per-prompt details (parquet), aggregate metrics (JSON)
  - Save full tuples: (prompt_id, prompt_text, response_text, pc_projections_dict, safety_score, capability_scores)
  - `src/utils/loader.py` — load results for analysis and visualization
  - Standard directory structure: `results/exp{N}_{name}/{config.yaml, metrics.json, details.parquet}`

- [ ] T2.7: Implement analysis utilities
  - `src/analysis/bootstrap.py` — BCa bootstrap CIs
  - `src/analysis/statistics.py` — Spearman correlation, permutation tests, BH-FDR correction
  - `src/analysis/lasso.py` — LASSO with nested CV for joint prediction
  - `src/analysis/effect_size.py` — Cohen's d with CI
  - Test: generate fake data, verify statistical functions produce correct results

- [ ] T2.8: Implement plotting module
  - `src/visualization/plots.py` — static matplotlib figures for paper
  - `src/visualization/interactive.py` — Plotly figures for dashboard
  - Initial plots: 3D PCA scatter, heatmap, steering curves, Pareto frontier
  - Test: generate one of each plot type from fake data

- [ ] T2.9: End-to-end smoke test
  - Write `src/experiments/smoke_test.py`
  - Full pipeline on Gemma 2 27B: extract → PCA → steer PC1 at strength 1.0 → eval safety (100 prompts) → eval capability (50 problems) → save results → generate plots
  - If this passes, infrastructure is ready
  - Document: total runtime, GPU memory usage, any issues

---

## Expected Outputs

- All `src/` modules implemented and individually tested
- Smoke test script that exercises the full pipeline
- Smoke test results in `results/smoke_test/`
- One example plot of each type in `results/smoke_test/figures/`

---

## Notes

- The smoke test is the gate for proceeding to Stage 3. If it doesn't pass cleanly, fix infrastructure before starting experiments.
- Judge API cost for smoke test: ~100 calls × ~500 tokens = ~50K tokens ≈ $0.02 on GPT-4.1-mini. Negligible.
- The full output tuple saving (T2.6) is critical for Viz 6 later. Don't skip it to save space — 15MB total for the whole project.
- For PCA validation (T2.2): the paper reports PC1-assistant-axis cosine sim > 0.71 at middle layer. If we get < 0.6, something is wrong with our extraction pipeline.
- TransformerLens hook names differ per model family. Document the exact hook for each Tier 1 model in a `configs/model_hooks.yaml`.
