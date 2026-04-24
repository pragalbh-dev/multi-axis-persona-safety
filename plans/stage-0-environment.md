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
  - Must support at minimum: Gemma 2 27B, Qwen 3 32B (thinking OFF), Gemma 4 31B-it (thinking ON+OFF), Qwen 3.6-27B dense judge — all in **fp8 or AWQ 4-bit** on 2× RTX 5090 with TP=2.
  - **fp8 support on 5090 Ada tensor cores is a hard requirement** — this is the 2-GPU hardware constraint's biggest consequence. vLLM has mature fp8 support; SGLang also supports fp8. Verify the specific engine build's fp8 path works on Ada (not just Hopper).
  - Pick one. Log: chosen engine, version, Python version, fp8 and AWQ verification status, why the other was rejected. Record in `CONVENTIONS.md` under "Inference engine" and "Python version".
  - Install via `uv` with pinned version in `pyproject.toml`.
  - **Out-of-scope for this stage:** Llama 3.3 70B and Qwen 3.6-35B-A3B MoE don't fit in 64 GB comfortably — they are Stage 7 Ext targets only. Engine choice doesn't need to optimize for them.

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

- [ ] **T0.4: Load Tier 1 subjects (2 models) — quantized per policy**
  - **Quant selection per subject** — follow CONVENTIONS "Quantization policy" preference order:
    1. Search HF for official fp8 checkpoint (e.g., `Qwen/Qwen3-32B-Instruct-FP8`, `neuralmagic/gemma-2-27b-it-FP8`).
    2. If absent, search for official or community AWQ 4-bit.
    3. If absent, check Unsloth's fp8/AWQ uploads.
    4. Last resort: self-calibrate AWQ using `autoawq` + 256 lmsys-chat-1m samples.
  - **Gemma 2 27B IT:** pick quant, record exact HF ID, VRAM, tensor_parallel_size (will be TP=2 on 2× 5090).
  - **Qwen 3 32B IT:** pick quant, record exact HF ID, VRAM, TP=2. Thinking mode OFF for fidelity to paper (record the disable mechanism in chosen engine).
  - Log exact HF IDs + quant format + provenance (official / community / Unsloth / self-calibrated) in `CONVENTIONS.md` under "Model IDs" AND a new entry in `plans/decisions.md` per subject.
  - **Llama 3.3 70B is NOT loaded here.** Deferred to Stage 7 Ext 9 pending GPU availability.

- [ ] **T0.5: Load Tier 2 core subject — Gemma 4 31B IT (quantized, both thinking modes)**
  - Same quant-selection protocol as T0.4 (official fp8 → official AWQ → Unsloth → self-calibrated AWQ).
  - Verify load on 2× 5090 at TP=2. Test both thinking ON and OFF toggles; record exact control mechanism (e.g., system-prompt token, config flag).
  - Log HF ID + quant format + thinking-toggle mechanism in `CONVENTIONS.md` and `decisions.md`.
  - **Qwen 3.6-35B-A3B MoE is Stage 7 Ext 1 only.** 35B at fp8 + MoE router state is marginal at 64 GB total; deferred.

- [ ] **T0.6: TransformerLens + nnsight smoke test**
  - `HookedTransformer.from_pretrained()` on Gemma 2 27B, extract residual stream from one forward pass
  - Same via nnsight
  - Document which backend works per model (nnsight for MoE likely)
  - Record exact hook-point name for "post-MLP residual stream at layer L" per model family (Gemma / Qwen / Llama / MoE) in `configs/model_hooks.yaml`
  - **Reasoning tokens (Gemma 4 31B thinking ON):** capture the exact hook point + token mask for extracting activations *during thinking tokens* AND *during answer tokens* separately. Document in `configs/model_hooks.yaml`. This is required for Stage 3's dual-mode extraction.
  - **lmsys-chat-1m norm reference:** for each Tier 1 + Tier 2 model, compute the average post-MLP residual stream norm at each layer on a sample from lmsys-chat-1m (paper uses n=18,777). Save to `data/cache/lmsys_norms/<model>.parquet`. Used later to scale steering vectors and random baselines per the paper's convention.

- [ ] **T0.7: Clone and audit starting codebases + pull paper's extraction-pipeline artifacts**
  - Clone `github.com/safety-research/assistant-axis`, `github.com/safety-research/persona_vectors`
  - Run their example notebooks / smoke tests
  - Audit and document in the Handoff:
    - What's reusable as-is (e.g., role vector extraction pipeline, pre-computed axes on HF)
    - What needs adapting (e.g., hard-coded model names)
    - What's missing — does either repo have: batched judge eval harness? multi-axis steering? capability eval integration? Note for Stage 2 agent.
  - Download pre-computed persona axes for Tier 1 models from HuggingFace.
  - **Also pull from assistant-axis repo (needed for Stage 3 Tier 2 extraction):**
    - **240 extraction questions** — the diverse prompts used to elicit role-expressing responses. Save to `data/paper_artifacts/extraction_questions.json`.
    - **5 default-Assistant system-prompt variants** (e.g., "You are a large language model", "Respond as yourself", + 3 others + the empty-system-prompt case). Save to `data/paper_artifacts/default_assistant_system_prompts.json`.
    - **Role-expression judge prompt template** (paper Appendix A, 3-label rubric). Save to `configs/role_expression_prompt.yaml`.
    - **Per-model rollouts / projection distributions IF released.** Check: does the HF release include the raw rollouts or the projection-distribution files for the Tier 1 models? If yes → cache them and skip Tier 1 rollout regeneration in Stage 3. If no → note in Handoff that Stage 3 T3.1 must regenerate rollouts for Tier 1 to populate the τ-calibration distribution.
    - **Per-model PC1 capping layer ranges** (paper Figure 10 / Appendix F for Llama + Qwen, Appendix for Gemma 2 27B). Save to `configs/paper_capping_ranges.yaml` so Stage 4 T4.0 can load Tier 1 configs without re-sweeping.

- [ ] **T0.8: Set up primary judge as a batch-processing step (Qwen 3.6-27B dense, quantized)**
  - **Not a persistent server.** Build a script that (a) loads the judge on both 5090s (TP=2), (b) streams a parquet of `(prompt, response)` rows through it in batches, (c) writes back labels + parses, (d) tears down.
  - **Quant:** same preference order (official fp8 → AWQ → Unsloth → self-calibrated). Log choice in `decisions.md`.
  - Chosen engine's offline/batch API (both vLLM and SGLang support this) preferred over HTTP server.
  - Config: thinking OFF by default, `max_input_len` / `max_output_len` from T0.2 judge values.
  - **TP=2 only** (we have 2 GPUs). data_parallel not available on this box.
  - Minimal test: feed 1,000 synthetic `(prompt, response)` rows, parse labels end-to-end, measure tokens/sec and steady-state GPU util (should be ≥90%).
  - Record batch size, TP config, tokens/sec, total-classification-time-for-1,100-prompts in `CONVENTIONS.md` under "Batch size & TP per model".

- [ ] **T0.9: Verify cross-check judge in the same load-batch-unload pattern (Gemma 4 31B-it, quantized)**
  - Use the same batch-processing script, swap model weights to quantized Gemma 4 31B-it.
  - Same quant-selection protocol (fp8 preferred). Can reuse the Gemma 4 31B config from T0.5 if the same checkpoint serves both subject + cross-check judge roles.
  - Thinking OFF default.
  - Test against the same 1,000-row synthetic input. Verify parseable labels.
  - Optional sanity check: thinking ON vs OFF on ~30 ambiguous prompts.
  - **Self-preference rule:** when Gemma 4 31B-it is the *subject* of an experiment, the orchestration script must skip the Gemma-as-judge pass on those prompts. Enforce this in the judge driver.

  **Co-location exception path is effectively closed on 2 GPUs.** All models use TP=2; there's no free GPU for a co-located judge. Phased topology (subject phase → judge phase → cross-check phase, each using full 2-GPU cluster) is strictly sequential.

  **Note:** the 200-sample judge validation against GPT-5.5 is deferred to **Stage 2 T2.4.5** — it needs the downloaded datasets (T0.10), a running subject model, and the finalized judge prompt (Stage 2 T2.0). Stage 0 only installs and smoke-tests the judge infrastructure.

- [ ] **T0.10: Download evaluation datasets**
  - Shah et al. persona-based jailbreak (1,100 prompts) — find the canonical source; assistant-axis repo may point to it
  - IFEval (541), MMLU Pro (1,400 subsample), GSM8k (1,000 subsample), EQ-Bench (171)
  - Save under `data/eval/<dataset>/`
  - Record exact HF IDs / source URLs in `CONVENTIONS.md` under "Eval dataset IDs"

- [ ] **T0.11: Baseline GPU util test for the phased pipeline (2× 5090 variant)**
  - End-to-end dry run of one phase sequence on dummy data, all models quantized per T0.4/T0.5/T0.8/T0.9:
    - Phase 1: load Gemma 2 27B quantized on both 5090s (TP=2), generate 1,000 responses to synthetic prompts, save to parquet, tear down.
    - Phase 2: load Qwen 3.6-27B judge quantized on both 5090s, classify those 1,000 responses, save labels, tear down.
    - Phase 3 (only if also verifying cross-check): load Gemma 4 31B-it quantized, classify a 200-row subset, save, tear down.
  - Measure per-phase: tokens/sec, steady-state GPU util (`nvidia-smi dmon` loop), wall-clock time, model load/unload overhead.
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
