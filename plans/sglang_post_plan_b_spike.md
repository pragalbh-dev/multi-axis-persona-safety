# SGLang `--forward-hooks` spike — post-Plan B follow-up

**Status:** Deferred from Plan B (April 26 deadline) to **post-deadline / Stage 7 candidate / first task of the April 27 → May 3 multi-subject sweep**.

**Why deferred (decision summary):** The feature is real and architecturally the right answer for fast steered/capped generation. But TP=4 on Gemma 4 31B-it is unvalidated by the SGLang team (cookbook documents TP=2 only on H200) and sm_120 fp8 has open issues. Spike cost ≈ 4–6 hr; Plan B steered savings ≈ 2 hr; ROI is positive only at multi-subject scale (where the steered-condition compute is ~16× larger and the spike pays back ~30+ saved hours).

---

## What SGLang's `--forward-hooks` actually is

Confirmed via primary sources (April 2026):

- **Real, documented, stable.** Defined in `python/sglang/srt/server_args.py` as `forward_hooks: Optional[List[dict[str, Any]]] = None`, CLI flag `--forward-hooks`, type `json_list_type`, help `"JSON-formatted forward hook specifications to attach to the model."` Implementation in `python/sglang/srt/model_executor/hook_manager.py` (`register_forward_hooks(...)`). Spec format: `{"target_modules": [glob], "hook_factory": "pkg.mod:fn", "name": str, "config": dict}`. Targets matched by `fnmatch` against `model.named_modules()`.
- **Landed via** [PR #13217](https://github.com/sgl-project/sglang/pull/13217) "Adding user defined hooks support"; renamed `--hooks` → `--forward-hooks` in [PR #13994](https://github.com/sgl-project/sglang/pull/13994) (merged 2025-11-26). Present in tagged releases v0.5.9, v0.5.10, v0.5.10.post1.
- **Hooks CAN modify activations during decode** by PyTorch contract. The factory returns a callable that's installed via `module.register_forward_hook(hook)`. PyTorch's documented semantics: if a hook returns a non-None tensor, that value replaces the module output. So our cap operator `h ← h - v · max(⟨h,v⟩ - τ, 0)` works as a hook return value. Continuous batching preserved.
- **Caveat — tuple outputs.** Some layer modules return `(hidden_states, residual)` tuples (RMSNorm-fused decoder layers in particular). Hook must handle both `tensor` and `tuple` cases and return the same shape it received.

## What's blocking immediate use

1. **Gemma 4 31B-it: TP=4 unvalidated.** [SGLang Cookbook: Gemma 4](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4) documents `google/gemma-4-31B-it` support but only at TP=2 on H200, TP=1 on MI300X / MI325X / MI355X. TP=4 on RTX 5090 (4× 32 GB Blackwell) is plausible but not in the validated configurations. Multimodal arch (`Gemma4ForConditionalGeneration`) means hook glob targets `model.language_model.layers.*` not `model.layers.*`.
2. **sm_120 fp8 broken.** [Issue #9233](https://github.com/sgl-project/sglang/issues/9233) and [#11576](https://github.com/sgl-project/sglang/issues/11576) document `fp8_blockwise_scaled_mm` not implemented for compute capability 12.0 as of SGLang 0.5.3, no resolution found in 0.5.10 release notes. Bf16 path is fine (which is what we run in core stages); but Stage 7 Ext 9 (Llama 70B at fp8) on SGLang would lose the validated path we pinned vLLM 0.19.1 + torch 2.10.0+cu128 for.
3. **Separate uv environment required.** SGLang pins its own torch + flashinfer versions; co-installing in the same env as `vllm==0.19.1` will conflict. Spike needs `uv venv .venv-sglang` with its own dependency tree.
4. **Hook factory authoring + numerical equivalence vs `external/assistant-axis::ActivationSteering`** are unproven against our harness. Stage 1 already wraps ActivationSteering in `src/steering/steerer.py::cap_and_steer / multi_axis_cap`; SGLang would need a parallel implementation behind the same API.

## Spike plan (4–6 hr realistic; 7–8 hr worst case)

| # | Step | Time | What |
|---|------|------|------|
| 1 | Separate uv env | 30 min | `uv venv .venv-sglang && source .venv-sglang/bin/activate && uv pip install sglang[all]==0.5.10`. Flashinfer wheel may need source build on Blackwell sm_120 (5–25 min). |
| 2 | Bf16 serve smoke (Gemma 2 27B, TP=4) | 30 min | `python -m sglang.launch_server --model-path google/gemma-2-27b-it --tp 4 --dtype bfloat16 --port 30000`. First load ~3 min; verify `/v1/chat/completions` returns. **Risk:** TP=4 on RTX 5090 unvalidated by SGLang team. |
| 3 | Write 4 hook factories in `tests/integration/sglang_hooks_smoke.py` | 60–90 min | (a) addition `h += λv`, (b) capping `h − v·max(⟨h,v⟩−τ, 0)`, (c) `cap_and_steer` (both ops, registered in order), (d) `multi_axis_cap` (one cap per direction, sequential). Each is a `hook_factory(config) -> callable`. Glob `target_modules: ["model.layers.22"]` (Gemma 2/Qwen 3 single-layer steer) and `["model.layers.18", ..., "model.layers.25"]` (8-layer cap range). Handle the tensor-vs-tuple output edge case. |
| 4 | Numerical equivalence vs HF reference | 60–90 min | Same 5 prompts through HF + `ActivationSteering` (cap at τ=25th percentile of synthetic projection distribution; addition at λ=±1), greedy decode at temperature=0. Run same prompts through SGLang + `--forward-hooks`. Compare token-by-token output **identity**, not just embeddings. Discrepancy modes to watch: hooks firing only on prefill, wrong tuple element captured, fnmatch glob hitting wrong module. |
| 5 | Decode-coverage assertion | 30 min | Generate 64 tokens with hooks installed. Add a counter inside the hook that bumps on each call. Assert `count == prefill_tokens + 64 × batch_size`. If decode count ≠ 64 × batch_size, the hook is prefill-only — disqualifies SGLang for our use case. |
| 6 | Gemma 4 31B sanity at TP=4 | 30–60 min | The "TP=4 unvalidated" risk. Same 4 hook patterns. Multimodal hook path is `model.language_model.layers.X`. If this fails but Gemma 2 works, SGLang is usable for Gemma 2 + Qwen 3 only; Gemma 4 stays HF for the multi-subject sweep. |

## Acceptance criteria for the spike

A successful spike must produce ALL of:

1. **Numerical token-level equivalence** vs HF + ActivationSteering on 5 prompts (greedy decode, temperature=0). Token IDs match exactly for the first 64 generated tokens under (a) addition λ=+1, (b) addition λ=−1, (c) capping τ=25th percentile.
2. **Decode-coverage** asserted (step 5 above): hook fires `prefill_tokens + 64 × batch_size` times in a 64-token decode test.
3. **All 4 hook patterns** work: addition, capping, `cap_and_steer`, `multi_axis_cap`.
4. **Gemma 4 31B-it at TP=4** loads and runs the same 4 patterns. (If this single criterion fails, SGLang is still usable for Gemma 2 + Qwen 3 + Gemma 4 thinking-OFF if those work — log per-subject status.)
5. **Throughput on bf16 27B Gemma 2 27B steered**: ≥3× speedup over HF + accelerate baseline on the same 100-prompt × 256-token workload.

If any of 1–3 fail: SGLang stays as a Stage 7 candidate; HF stays as the steered backend for the post-deadline sweep. If only 4 or 5 fail: log per-subject SGLang eligibility, use HF for the failing subjects.

## Migration plan (if spike succeeds)

The Plan B / Stage 2 architecture treats backend choice as a runtime arg on the work-module. Adding SGLang would mean:

1. New `src/evaluation/run_subject_rollouts.py --backend sglang` branch wrapping the SGLang client with the `--forward-hooks` JSON config built from the same `SteeringConfig` Stage 1 already defines.
2. New `src/steering/sglang_hook_factories.py` exposing the 4 factories that match `external/assistant-axis::ActivationSteering` semantics 1:1. Tests pin numerical equivalence (greedy decode, top-k token IDs).
3. `configs/inference_runtime.yaml` gets a `backend: sglang` profile with `--forward-hooks` JSON and tuned `--max-running-requests`, `--mem-fraction-static`. Run grid search for SGLang the same way we did for vLLM.
4. **Per-subject opt-in.** Subjects that pass spike criterion 4 can use SGLang; failures stay HF. The phased orchestrator (`src/evaluation/full.py::eval_full`) reads `cfg.steered_backend ∈ {hf, sglang}` from the experiment config.
5. **No change to Stage 1/2 module APIs.** `cap_and_steer`, `multi_axis_cap`, `ActivationSteering`, `eval_safety`, `eval_full` all keep their signatures. SGLang is a new backend behind the `run_subject_rollouts` work-module, period.

## Cost / benefit summary

| | Plan B (single subject, 500 prompts) | Post-deadline sweep (4 subjects × 1100 prompts × 2 datasets) |
|---|---|---|
| HF steered runtime | ~3 hr | ~95 hr |
| SGLang steered runtime (if spike works) | ~50 min | ~28 hr |
| **Net savings vs HF** | **2 hr 10 min** | **~67 hr** |
| Spike cost | 4–6 hr | 4–6 hr |
| **ROI** | **Negative (-2 to -4 hr)** | **Strongly positive (+60+ hr)** |

## Sources

- [`server_args.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) — `forward_hooks` argument
- [`hook_manager.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/hook_manager.py) — `register_forward_hooks` implementation
- [PR #13217](https://github.com/sgl-project/sglang/pull/13217) — initial landing
- [PR #13994](https://github.com/sgl-project/sglang/pull/13994) — `--hooks` → `--forward-hooks` rename
- [Issue #3266](https://github.com/sgl-project/sglang/issues/3266) — original feature request (closed)
- [SGLang Cookbook: Gemma 4](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4) — TP=2 documented, TP=4 not
- [PR #21952](https://github.com/sgl-project/sglang/pull/21952) — Gemma 4 model support
- [Issue #9233](https://github.com/sgl-project/sglang/issues/9233) — fp8 block on sm_120
- [Issue #11576](https://github.com/sgl-project/sglang/issues/11576) — SGLang 0.5.3 fp8 broken on RTX 5090
- [Server Arguments doc](https://docs.sglang.io/advanced_features/server_arguments.html)
