# Stage 0: Environment Setup

**Objective:** Install tools, pick versions, download models, set up self-hosted judges, and verify the full pipeline runs on 4x RTX 5090 at ≥90% GPU util. Zero experiment code — pure setup, audits, and smoke testing.

---

## Required inputs

- `../CLAUDE.md` — Key Decisions (models, evaluation, inference, tooling, data)
- `CONVENTIONS.md` — locked decisions + sections under "Decide and log" that you fill in during this stage

No prior stage. You are the first.

---

## Prerequisites

- 4× RTX 5090 accessible via SSH (`synaptic@synaptic-5090` on Tailscale)
- HuggingFace token in `.env` with access to gated models (Gemma, Llama)
- `~/obsidian-vault/` symlink working on the GPU box

---

## Completion criteria

- Env installed with pinned versions
- All 3 Tier 1 + 2 Tier 2 subject models + 1 primary judge (Qwen 3.6-27B) loaded and verified
- Judge server serves batched requests at ≥90% GPU util, cross-check judge (Gemma 4 31B-it) verified
- Starting codebases (`safety-research/assistant-axis`, `safety-research/persona_vectors`) audited — documented what's reusable
- Stage 0 → Stage 1 Handoff written in `progress.md`
- All "Decide and log" entries in `CONVENTIONS.md` relevant to this stage filled in

---

## Tasks

- [ ] **T0.1: Decide and install inference engine (vLLM vs SGLang)**
  - Check release notes / model support matrix for both engines at the current version.
  - Must support at minimum: Gemma 2 27B, Qwen 3 32B (thinking OFF), Llama 3.3 70B, Gemma 4 31B-it (thinking ON+OFF), Qwen 3.6-35B-A3B MoE (thinking ON+OFF), Qwen 3.6-27B dense judge.
  - Pick one. Log: chosen engine, version, Python version, why the other was rejected. Record in `CONVENTIONS.md` under "Inference engine" and "Python version".
  - Install via `uv` with pinned version in `pyproject.toml`.

- [ ] **T0.2: Token-distribution audit across all eval datasets**
  - Tokenize a 200-sample subset from each: Shah et al. 1,100 persona jailbreaks, IFEval, MMLU Pro, GSM8k, EQ-Bench.
  - Use the tokenizer of each target model (distributions differ across tokenizers).
  - Compute p50 / p95 / p99 input lengths, estimate required output length per task.
  - Set `max_input_len` / `max_output_len` per (task, model) in `configs/` and log to `CONVENTIONS.md` under "Max input/output lengths per task".
  - Rationale: these sizes directly drive KV cache memory allocation. Too large = wasted VRAM / fewer batches; too small = truncation / retries.

- [ ] **T0.3: Set up Python env with uv**
  - `uv venv`, Python pinned per T0.1 decision
  - `pyproject.toml` with: torch, transformers, accelerate, bitsandbytes, transformer-lens, nnsight, chosen inference engine, ruff, mypy, pytest, plotly, dash, pandas, pyarrow, scikit-learn, python-dotenv, pyyaml
  - Lockfile committed (`uv lock`)
  - Create `.env.example` with required key names (HF_TOKEN, optionally WANDB_API_KEY)
  - Verify CUDA available on all 4 GPUs

- [ ] **T0.4: Load all Tier 1 models**
  - Gemma 2 27B IT — record exact HF ID, VRAM, tensor_parallel_size
  - Qwen 3 32B IT — thinking mode OFF for fidelity to paper; record how to disable in chosen engine
  - Llama 3.3 70B IT — likely needs fp8 or int8 on 4× 5090 (128 GB total); test bf16-with-offloading vs fp8 for throughput
  - Log exact HF IDs in `CONVENTIONS.md` under "Model IDs"

- [ ] **T0.5: Load Tier 2 models**
  - Gemma 4 31B IT — verify load; test both thinking ON and OFF toggles; record exact control mechanism (e.g., `<|think|>` token in system prompt)
  - Qwen 3.6-35B-A3B MoE — verify MoE load; test thinking ON/OFF; identify residual stream hook points (MoE has router state too — note which hooks give us persona signal)
  - Log exact HF IDs and thinking-toggle mechanism in `CONVENTIONS.md`

- [ ] **T0.6: TransformerLens + nnsight smoke test**
  - `HookedTransformer.from_pretrained()` on Gemma 2 27B, extract residual stream from one forward pass
  - Same via nnsight
  - Document which backend works per model (nnsight for MoE likely)
  - Record exact hook-point name for "post-MLP residual stream at layer L" per model family (Gemma / Qwen / Llama / MoE) in `configs/model_hooks.yaml`

- [ ] **T0.7: Clone and audit starting codebases**
  - Clone `github.com/safety-research/assistant-axis`, `github.com/safety-research/persona_vectors`
  - Run their example notebooks / smoke tests
  - Audit and document in the Handoff:
    - What's reusable as-is (e.g., role vector extraction pipeline, pre-computed axes on HF)
    - What needs adapting (e.g., hard-coded model names)
    - What's missing — does either repo have: batched judge eval harness? multi-axis steering? capability eval integration? Note for Stage 2 agent.
  - Download pre-computed persona axes for Tier 1 models from HuggingFace.

- [ ] **T0.8: Set up primary judge as a batch-processing step (Qwen 3.6-27B dense)**
  - **Not a persistent server.** Build a script that (a) loads the judge on all 4 GPUs, (b) streams a parquet of `(prompt, response)` rows through it in batches, (c) writes back labels + parses, (d) tears down.
  - Chosen engine's offline/batch API (both vLLM and SGLang support this) is preferred over an HTTP server for the default path.
  - Config: thinking OFF by default, `max_input_len` / `max_output_len` from T0.2 judge values.
  - Try TP=4 first, then TP=2 × data_parallel=2 — pick whichever hits higher throughput for the judge's input/output shape.
  - Minimal test: feed 1,000 synthetic `(prompt, response)` rows, parse labels end-to-end, measure tokens/sec and steady-state GPU util (should be ≥90%).
  - Record batch size, TP/DP config, tokens/sec, total-classification-time-for-1,100-prompts in `CONVENTIONS.md` under "Batch size & TP per model".

- [ ] **T0.9: Verify cross-check judge in the same load-batch-unload pattern (Gemma 4 31B-it)**
  - Use the same batch-processing script, swap model weights to Gemma 4 31B-it.
  - Thinking OFF default.
  - Test against the same 1,000-row synthetic input. Verify parseable labels.
  - Optional sanity check: thinking ON vs OFF on ~30 ambiguous prompts — if ON materially changes labels, flag for Stage 2 judge-prompt design.
  - **Self-preference rule:** when Gemma 4 31B-it is the *subject* of an experiment, the orchestration script must skip the Gemma-as-judge pass on those prompts. Enforce this in the judge driver, not in manual discipline.
  - **Exception path (keep for later):** if a subject model is small enough to leave ≥2 GPUs free during its phase (e.g., Gemma 2 27B bf16 on 2 GPUs), a co-located judge endpoint is allowed as an optimization. Not the default — document the optimization path but don't build an always-on server in Stage 0.

- [ ] **T0.10: Download evaluation datasets**
  - Shah et al. persona-based jailbreak (1,100 prompts) — find the canonical source; assistant-axis repo may point to it
  - IFEval (541), MMLU Pro (1,400 subsample), GSM8k (1,000 subsample), EQ-Bench (171)
  - Save under `data/eval/<dataset>/`
  - Record exact HF IDs / source URLs in `CONVENTIONS.md` under "Eval dataset IDs"

- [ ] **T0.11: Baseline GPU util test for the phased pipeline**
  - End-to-end dry run of one phase sequence on dummy data:
    - Phase 1: load Gemma 2 27B on all 4 GPUs, generate 1,000 responses to synthetic prompts, save to parquet, tear down.
    - Phase 2: load Qwen 3.6-27B judge on all 4 GPUs, classify those 1,000 responses, save labels, tear down.
    - Phase 3 (only if also verifying cross-check): load Gemma 4 31B-it, classify a 200-row subset, save, tear down.
  - Measure per-phase: tokens/sec, steady-state GPU util (`nvidia-smi dmon` loop), wall-clock time, and model load/unload overhead.
  - **Acceptance: ≥90% steady-state GPU util during each generate/classify phase.** If load/unload overhead > 10% of phase duration, flag for Stage 2 — we may need to batch multiple experiments' judging into one judge load.
  - Record per-phase numbers in `CONVENTIONS.md` under "Batch size & TP per model".

- [ ] **T0.12: Write Stage 0 → Stage 1 Handoff**
  - Append the Handoff block to `progress.md`
  - Include: exact model IDs, engine/version, Python version, max_input/output per task, batch size per model, what's reusable from assistant-axis and persona_vectors, any model-specific gotchas (e.g., "Qwen 3.6 MoE routing is non-deterministic without seed"), remaining open items
  - Update `CLAUDE.md` "Current State" to Stage 1

---

## Expected Outputs

- Working `uv` venv with pinned `pyproject.toml` + lockfile
- All 5 subject models + 2 judge models verified loadable
- Judge server running, verified at ≥90% GPU util
- Cross-check judge verified
- Starting codebases cloned and audited (audit notes in Handoff)
- Eval datasets downloaded
- `configs/model_hooks.yaml` with hook point names per model
- `CONVENTIONS.md` "Decide and log" sections filled in
- Stage 0 → Stage 1 Handoff written

---

## Notes

- Llama 3.3 70B at bf16 = ~140 GB. With 4× 5090 (128 GB) we need fp8/int8 or offloading. Test both for throughput; fp8 is usually fastest on 5090.
- Qwen 3 and Qwen 3.6 families default to thinking ON. Tier 1 Qwen 3 needs thinking OFF (reproduction fidelity); Tier 2 Qwen 3.6 MoE needs BOTH modes tested.
- Gemma 4 31B-it plays two roles: Tier 2 subject *and* cross-check judge. Use separate server instances so runs don't interfere.
- The assistant-axis repo has pre-computed axes for Tier 1 — use those rather than recomputing. Tier 2 still needs extraction from scratch (Stage 7 Ext 1).
- `trust_remote_code=True` may be required for Qwen 3 and Qwen 3.6. Document in `CONVENTIONS.md`.
- Record any inference-engine quirks you hit so Stage 2 implementers don't rediscover them.
