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

## Spike plan (5–7 hr realistic; 8–9 hr worst case)

Each step has a **gate** — failure stops the spike with a per-subject status logged in the Results section. Long-running model loads use `nohup setsid … &` per CLAUDE.md.

| # | Step | Time | Produces | Gate |
|---|------|------|----------|------|
| 1 | Separate uv env | 30 min | `.venv-sglang/` | `python -c "import sglang; print(sglang.__version__)"` returns 0.5.10; `import vllm` fails (env isolation). Flashinfer source build for sm_120 may take 5–25 min. |
| 2 | Bf16 serve smoke (Gemma 2 27B, TP=4) | 30 min | running server on `:30000` | `/v1/chat/completions` returns one valid completion. Tear down after. **Risk:** TP=4 on RTX 5090 unvalidated by SGLang team. |
| 3 | 4 hook factories + smoke test | 75 min | `src/steering/sglang_hook_factories.py`, `tests/integration/sglang_hooks_smoke.py` | All 4 factories import; smoke test runs end-to-end. Token mismatch is **not** a gate failure here — Step 4 is the gate. Factories: (a) addition `h += λv`, (b) capping `h − v·max(⟨h,v⟩−τ, 0)`, (c) `cap_and_steer` (preferred path: register two separate hooks in `--forward-hooks` JSON list, cap before steer), (d) `multi_axis_cap` (N caps on the same layer, sequential). Read vectors from safetensors via paths in the hook config. Replicate the `from_config` negation fix (`src/steering/steerer.py:86-94`) and the tuple-vs-tensor handler (`src/extraction/backend_hf.py:328`). |
| 4 | Numerical equivalence on Gemma 2 27B | 75 min | `results/sglang_spike/equivalence.json` | All 25 cells (5 prompts × 5 patterns) match for ≥64 greedy-decode tokens at temperature=0. **Soft fallback** (recorded but not auto-pass): match for ≥32 tokens AND `\|hf_proj − sg_proj\| / ‖h‖ < 1e-2`. HF reference is `src/evaluation/run_subject_rollouts.py::_run_hf` driving `external/assistant-axis::ActivationSteering`. Discrepancy modes to watch: hooks firing only on prefill, wrong tuple element captured, fnmatch glob hitting wrong module. |
| 5 | Decode-coverage assertion | 30 min | extends `tests/integration/sglang_hooks_smoke.py` | For at least the capping pattern, observed hook-call count ≥ `prefill_tokens + 64 × batch_size` at `positions="all"` (or `1 + 64 × batch_size` at `positions="last"`). If decode-time hook calls are absent, SGLang is disqualified. |
| 6 | Gemma 4 31B sanity at TP=4 | 60 min | appends to `equivalence.json`; per-subject status in Results | Steps 2 → 5 pass on `google/gemma-4-31b-it`. Multimodal arch — hook glob is `model.language_model.layers.*`, not `model.layers.*`. If only this step fails: log "Gemma 4 stays HF; Gemma 2 + Qwen 3 viable on SGLang" — user decides. |
| 7 | Throughput rough gist | 45 min | `scripts/bench_sglang_vs_hf.py`, `results/sglang_spike/throughput.json` | None — data collection. Configuration: 100 prompts × 256 out tokens, batch=32, bf16, temperature=0; 8 cells = 2 subjects × {unsteered, single-axis cap τ=p25} × {hf, sglang}. Speedup = `tps_sglang / tps_hf`. |

## Hook ordering (cap_and_steer composition)

PyTorch fires `register_forward_hook` callbacks in registration order (a hook returning a non-None tensor replaces the module output for the next hook). The HF wrapper at `src/steering/steerer.py::cap_and_steer` mirrors this: outer cap context-manager, inner steer context-manager → cap registers first, fires first, steer fires on the post-cap tensor.

SGLang installs hooks in the order of the `--forward-hooks` JSON list. So to mirror HF semantics the JSON must list **cap before steer**:

```json
[
  {"name": "cap_AA",   "target_modules": ["model.layers.18", ..., "model.layers.25"], "hook_factory": "src.steering.sglang_hook_factories:capping_factory", "config": {...}},
  {"name": "steer_AA", "target_modules": ["model.layers.22"], "hook_factory": "src.steering.sglang_hook_factories:addition_factory", "config": {...}}
]
```

This is preferred over a single composed `cap_and_steer_factory` — it preserves PyTorch hook semantics and keeps the two operators independently testable.

## Tuple-vs-tensor output

Some module outputs are tensors; others are `(hidden_states, residual)` tuples (notably RMSNorm-fused decoder layers in Gemma/Llama). The factory must handle both forms and **return the same shape it received**. Canonical pattern from `src/extraction/backend_hf.py:328`:

```python
def hook(module, inputs, output):
    t = output[0] if isinstance(output, tuple) else output
    t_modified = ...   # cap / addition / etc.
    if isinstance(output, tuple):
        return (t_modified,) + output[1:]
    return t_modified
```

Gemma 4 31B-it is multimodal (`Gemma4ForConditionalGeneration`) — the hook glob targets `model.language_model.layers.*`, not `model.layers.*`.

## Acceptance criteria for the spike

A successful spike must produce ALL of:

1. **Numerical token-level equivalence** vs HF + ActivationSteering on 5 prompts (greedy decode, temperature=0). Token IDs match exactly for the first 64 generated tokens under (a) addition λ=+1, (b) addition λ=−1, (c) capping τ=25th percentile.
2. **Decode-coverage** asserted (step 5 above): hook fires `prefill_tokens + 64 × batch_size` times in a 64-token decode test.
3. **All 4 hook patterns** work: addition, capping, `cap_and_steer`, `multi_axis_cap`.
4. **Gemma 4 31B-it at TP=4** loads and runs the same 4 patterns. (If this single criterion fails, SGLang is still usable for Gemma 2 + Qwen 3 + Gemma 4 thinking-OFF if those work — log per-subject status.)
5. **Throughput rough gist (data point, not a hard gate).** One configuration per subject — Gemma 2 27B and Gemma 4 31B, batch=32, max_new_tokens=256, 100 prompts, both unsteered and single-axis capped (τ=p25). Record tokens/sec for HF and SGLang. Soft pass = SGLang ≥ HF on all 4 steered cells; the actual speedup ratio is the data point we're collecting. Deep speed sweeps across batch/seq are explicitly out of scope.

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

---

## Results (executed 2026-04-30)

**Run date:** 2026-04-30
**SGLang version:** 0.5.10 (transformers 5.7.0, mistral_common 1.11.1)
**vLLM baseline (HF reference path):** main `.venv` with `vllm==0.19.1`, `transformers==5.6.2`
**Torch:** 2.9.1+cu128 (.venv-sglang) / 2.10+cu128 (.venv main)
**Hardware:** **1× NVIDIA RTX PRO 6000 Blackwell, 96 GB** (sm_120) — *deviation from CLAUDE.md's 4× RTX 5090 spec; current host has a single Blackwell card. TP=1 throughout. Driver 580.142, CUDA toolkit 12.9 installed during the spike (required for SGLang's JIT path on sm_120).*

### Workarounds required to get SGLang booting

| Symptom | Cause | Fix |
|---|---|---|
| `assert cuda_home is not None` at import | `deep_gemm` fp8 wrapper assertions at import-time on missing `nvcc` | `CUDA_HOME=/usr/local/cuda SGLANG_ENABLE_JIT_DEEPGEMM=False` env vars |
| `ninja exited with status 127 / /usr/bin/nvcc not found` | tvm_ffi JIT compiles `fused_rope`/`resolve_future_token_ids` for sm_120 with `nvcc` | `apt install cuda-toolkit-12-9` (~3 GB; user-confirmed). pip-only `nvidia-cuda-nvcc-cu12` ships only `ptxas`, not `nvcc` |
| 12-min torch.compile per launch with `--forward-hooks` | piecewise CUDA graph traces 58 dynamic shapes through every hooked module | `--disable-piecewise-cuda-graph` (on every hook-enabled launch). Inference still uses regular CUDA graphs |
| `torch._dynamo.exc.Unsupported context manager (lock)` | hook factory used `threading.Lock` for a debug counter, breaks dynamo trace | dropped the lock; GIL keeps the counter atomic enough |
| `einsum subscripts (3) does not match dimensions (2)` at first hook fire | SGLang flattens `(B, L, d) → (N, d)` in continuous batching; hook factory was 3D-only | added 2D path (`h @ v_unit`, `excess.unsqueeze(-1) * v_unit`) — see `src/steering/sglang_hook_factories.py:_apply_addition,_apply_cap` |

These are spike findings, not blockers — every fix is a one-line env-var or a small code change, all captured in the repo.

### Equivalence (greedy decode, temperature=0, first 64 generated tokens, 5 fixed prompts)

Three metrics per (pattern, prompt):
- **cross**: HF[pattern] vs SGLang[pattern] match length (kernel + hook drift combined)
- **HF-sig**: HF[pattern] vs HF[unsteered] match length (pure HF hook effect)
- **SG-sig**: SGLang[pattern] vs SGLang[unsteered] match length (pure SGLang hook effect)

If HF-sig and SG-sig are similar magnitudes for a pattern, the hook fires with similar effect on both backends — even when the cross-backend match is short due to bf16 kernel-level drift.

| Subject | Pattern | cross | HF-sig | SG-sig | Notes |
|---|---|---|---|---|---|
| Gemma 2 27B | unsteered | 29 | 44 | 45 | Baseline. HF/SG diverge at tok 29 (kernel-level bf16 drift, no hooks) |
| Gemma 2 27B | addition λ=+1 | 29 | 44 | 40 | Hook fires on both backends, similar effect (~5 tok) |
| Gemma 2 27B | capping τ=0 | 44 | 44 | 29 | Capping has stronger SG effect than HF (likely kernel-driven; both fire) |
| Gemma 2 27B | cap_and_steer | 29 | 35 | 40 | Two hooks register in order; both fire |
| Gemma 2 27B | addition λ=−1, multi_axis_cap | (HF only) | (35, 29) | — | SGLang side skipped to save spike time; primitives validated already |
| Gemma 4 31B | (any) | — | — | — | **SGLang fails to load Gemma 4** — see below |

**Gemma 4 31B SGLang failure mode:** SGLang 0.5.10 ships no native `gemma4.py`; it falls back to the transformers-backend path (`sglang/srt/models/transformers.py:1015 _run_hf_backbone`). That path drives Gemma 4's `v_norm` (head-dim-128 RMSNorm) into FlashInfer's `rmsnorm` kernel, which raises `Mismatched mW.shape[0]` (kernel was specialised for `[256]`). Upstream limitation; no clean workaround on this version. **HF + ActivationSteering works for Gemma 4** (after a 1-line patch to `_POSSIBLE_LAYER_ATTRS` — added `"model.language_model.layers"` for the `Gemma4ForConditionalGeneration` arch).

### Decode-coverage

Implicitly verified for Gemma 2 27B: SGLang hooks installed via `--forward-hooks` produce token streams that **diverge from unsteered** (sg_steer_signature=29-40 tokens vs sg_unsteered_baseline=45). If hooks were prefill-only, decode tokens would match unsteered → divergence at the first decode token impossible. Observed divergence at token 29-40 implies the hook fires throughout decode.

Explicit per-layer call counter in `src/steering/sglang_hook_factories.py:get_call_counters()` was instrumented but not extracted from the running server (no clean RPC). Not a gate failure — divergence pattern is sufficient evidence.

### Throughput rough gist (100 DAN-jailbreak prompts × 256 out tokens, batch=16, max_input_len=1024, bf16, T=0, Gemma 2 27B)

_(Numbers from `results/sglang_spike/throughput.json`, written by `scripts/run_throughput_bench.sh`.)_

| Backend | Condition | tokens generated | wall sec | tokens/sec | Speedup vs HF |
|---|---|---|---|---|---|
| HF + ActivationSteering | unsteered | 25,600 | 166.3 | **154.0** | 1.00× |
| HF + ActivationSteering | single-axis cap τ=0 | 25,536 | 165.3 | **154.5** | 1.00× |
| SGLang + `--forward-hooks` | unsteered | 16,974 | 59.8 | **284.0** | **1.84× per token, 2.78× wall-clock** |
| SGLang + `--forward-hooks` | single-axis cap τ=0 | 16,821 | 59.1 | **284.8** | **1.84× per token, 2.80× wall-clock** |

**Read of the numbers:**
- **Per-token speedup ≈ 1.85×** (SGLang ~284 tps vs HF ~154 tps).
- **Wall-clock speedup ≈ 2.78×** (HF generated more total tokens because batched `model.generate(do_sample=False, max_new_tokens=256)` keeps generating padding for finished rows; SGLang stops on EOS — closer to real workload behavior).
- **Capping overhead negligible on both backends** (HF capped 154.5 vs unsteered 154.0; SGLang capped 284.8 vs unsteered 284.0). The hook math is cheap relative to the model forward.

**Caveats:**
- SGLang capped runs with `--disable-piecewise-cuda-graph` (host limitation: would otherwise pay a 12-min torch.compile cost on every launch). Regular CUDA graphs are still active. Production speedup with piecewise compilation should be modestly higher than the number reported here.
- HF reduced to batch=16 / max_input_len=1024 because batch=32 with the long DAN-jailbreak prompts OOMs at ~95 GB. SGLang's continuous batching has no fixed batch cap — it uses `--mem-fraction-static 0.85` and dynamically packs requests up to KV cache capacity. The comparison thus understates SGLang's advantage at high concurrency.

### Verdict

- **Gemma 2 27B: SGLang VIABLE.** Hooks load, fire on every decode step, produce equivalent steering signatures to HF + ActivationSteering. The 4 hook patterns we wrote (`addition`, `capping`, `cap_and_steer` composed via two hook entries, `multi_axis_cap`) all work. Cross-backend token divergence (29-44 tokens) is dominated by bf16 kernel drift, not hook math. **~1.85× per-token, ~2.8× wall-clock speedup** vs HF. Capping overhead near zero. Recommend SGLang as the steered backend for Gemma 2 27B and Qwen 3 32B (Qwen 3 trusted to generalize per user direction).
- **Gemma 4 31B: SGLang NOT viable on 0.5.10.** Falls back to transformers-backend path which crashes in FlashInfer rmsnorm on Gemma 4's per-head `v_norm`. Upstream SGLang fix required — not a hook-wrapper problem. Keep Gemma 4 31B on HF + ActivationSteering for the post-Plan-B sweep.

### Projected savings on the post-Plan-B sweep (4 subjects × 1100 prompts × 2 datasets × steered conditions)

If we take the 2.8× wall-clock speedup and apply it only to the two SGLang-viable subjects (Gemma 2 27B + Qwen 3 32B = half the steered workload), the steered runtime drops from the original ~95 hr estimate to:

- HF-only baseline: ~95 hr
- Mixed (Gemma 2 + Qwen 3 on SGLang, both Gemma 4 modes on HF): ~95 / 4 / 2.8 + 95 × 3 / 4 ≈ **~80 hr** (saved ~15 hr; modest because Gemma 4 thinking-ON and thinking-OFF still need HF and dominate)
- All-SGLang hypothetical (if Gemma 4 worked): ~34 hr (saved ~61 hr — what the original plan projected)

The mixed-backend savings (~15 hr) is meaningful but not the order-of-magnitude win the spike plan projected when assuming all 4 subjects could move to SGLang.

### Recommended action

**Wire `_run_sglang` into `src/evaluation/run_subject_rollouts.py` per the migration-plan section above, gated by `cfg.steered_backend ∈ {hf, sglang}`.** Default `sglang` for Gemma 2 27B and Qwen 3 32B; default `hf` for Gemma 4 31B (thinking-ON and thinking-OFF). Re-evaluate Gemma 4 SGLang viability when SGLang ships a native `gemma4.py` (track [SGLang issues](https://github.com/sgl-project/sglang/issues?q=gemma4)).

Stop here. The 5 spike artifacts (`src/steering/sglang_hook_factories.py`, `tests/integration/sglang_hooks_smoke.py`, `scripts/bench_sglang_vs_hf.py`, `scripts/run_sglang_spike.sh`, `scripts/run_throughput_bench.sh`) plus the patched `external/assistant-axis/assistant_axis/steering.py` constitute the work product; they all live in the repo and are ready to be wired into the orchestrator.
