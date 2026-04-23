# Stage 0: Environment Setup

**Objective:** Install all tools, download models, verify everything runs on 4x RTX 5090. Zero code written — this is purely setup and smoke testing.

**Prerequisites:** None.

**Completion criteria:** Can load each Tier 1 model, extract residual stream activations, run a single PCA, and call the judge API successfully.

---

## Tasks

- [ ] T0.1: Create Python environment
  - Python 3.11+ virtual environment (conda or venv)
  - Install core deps: torch, transformers, accelerate, bitsandbytes
  - Verify CUDA is available on all 4 GPUs

- [ ] T0.2: Install TransformerLens
  - `pip install transformer-lens`
  - Load Gemma 2 27B IT in TransformerLens with `HookedTransformer.from_pretrained()`
  - Verify: extract residual stream activations from one forward pass
  - Note the hook point names for post-MLP residual stream

- [ ] T0.3: Install nnsight as backup
  - `pip install nnsight`
  - Load same model via nnsight
  - Verify: same activation extraction works
  - Document which tool is faster/more convenient per model

- [ ] T0.4: Load all Tier 1 models
  - Gemma 2 27B IT — verify loads, note GPU allocation
  - Qwen 3 32B IT — verify loads (may need `trust_remote_code=True`)
  - Llama 3.3 70B IT — verify loads (will need multi-GPU, possibly fp8/int8)
  - Document VRAM usage and optimal parallelism strategy per model

- [ ] T0.5: Clone and test starting codebases
  - Clone `github.com/safety-research/assistant-axis`
  - Clone `github.com/safety-research/persona_vectors`
  - Run their smoke tests / example notebooks
  - Download pre-computed persona axes from HuggingFace
  - Verify: can load pre-computed role vectors and reproduce their PCA

- [ ] T0.6: Set up judge API
  - Get API key for GPT-4.1-mini (or Gemini 2.5 Flash)
  - Write a minimal test: send 5 (prompt, response) pairs, get safety scores back
  - Verify: response format, latency, cost per call
  - Document: chosen judge model, API endpoint, estimated cost

- [ ] T0.7: Download evaluation datasets
  - Shah et al. persona-based jailbreak dataset (1,100 prompts)
  - IFEval, MMLU Pro, GSM8k, EQ-Bench datasets
  - Verify: can load each, correct format, correct number of examples
  - Save to `data/eval/`

- [ ] T0.8: Create .gitignore
  - Ignore: `data/`, `results/`, `*.pt`, `*.safetensors`, `*.bin`, `__pycache__/`, `.env`, `notebooks/*.ipynb_checkpoints`
  - Keep: `configs/`, `src/`, `plans/`, `report/`, `dashboard/`

---

## Expected Outputs

- Working Python environment with all dependencies
- All 3 Tier 1 models verified loadable
- Pre-computed persona axes downloaded
- Judge API tested
- Eval datasets downloaded to `data/eval/`
- `.gitignore` in place
- Notes in this file on any issues/workarounds discovered

---

## Notes

- Llama 3.3 70B at BF16 = ~140GB. With 4x 5090 (128GB) we need fp8 or int8 quantization, or offloading. Test both approaches and pick the faster one.
- Qwen 3 32B may need `trust_remote_code=True` and potentially a specific transformers version.
- The assistant-axis repo has pre-computed axes on HuggingFace — we should use these for Tier 1 models rather than recomputing from scratch. Recomputation is only needed for Tier 2 (new models).
- TransformerLens hook points vary by model family. Document the exact hook point name for "post-MLP residual stream at layer L" for each model.
