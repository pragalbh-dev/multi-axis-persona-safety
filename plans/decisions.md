# Decisions Ledger

Append-only log of **unplanned decisions** — any choice made during stage execution that was NOT in the pre-written stage plan.

**What belongs here:**
- Picking a specific integer / hyperparameter when the plan said "something around X" (e.g., "extraction layer L* landed at 30, not the paper's reported 32 — difference of 2, log why").
- Resolving an ambiguity in the paper's convention (e.g., paper says "25th percentile" — of which distribution? of which activations? if we chose differently from the plan's default).
- Picking between two viable library versions or implementations.
- Small scope cuts taken under time pressure ("skipped the cross-check judge on 50 prompts because judge server was GPU-starved").
- Anything you'd want a future agent (or the user) to be able to audit, question, and potentially reverse.

**What does NOT belong here:**
- Progress updates (those go in `progress.md`).
- Task completion checkboxes (those go in the stage plan).
- Decisions already specified in the stage plan or `CONVENTIONS.md` (those were pre-planned — no need to log).
- Transient state during implementation (only the final decision goes here).

**Rule of thumb:** if you caught yourself saying "the plan didn't cover this, so I'm going with X" — that's a decision. Log it.

---

## Template

Copy this block, fill in, append at the **end** of this file. Do not edit earlier entries.

```markdown
## [YYYY-MM-DD HH:MM] Stage N / T{k.m} — {short title}

**Decision:** {what was decided in 1-2 sentences}

**Alternatives considered:**
- Option A — {why rejected}
- Option B — {why rejected}

**Reason:** {why this option won. Cite the constraint that forced the choice.}

**Source:** {pick one or more}
- Paper line / appendix reference (e.g., "paper line 691" or "Appendix D.2.2")
- File path (e.g., "configs/paper_capping_ranges.yaml")
- External URL
- User instruction ("user said in chat on 2026-04-24 to prefer X over Y")
- Own judgment ("no source; judgment call because …")

**Reversibility:** {high / medium / low}
- **High** — decision can be flipped by changing one config value and re-running one task.
- **Medium** — flipping requires re-running a stage's worth of work (hours of GPU).
- **Low** — flipping invalidates downstream experiments or requires restarting multiple stages.

**How to revert:** {concrete steps. Which file(s) to edit? Which artifacts need regeneration? Any handoff blocks to update in `progress.md`?}

**Downstream dependencies:** {what later stages / tasks are built on top of this decision?}
```

---

## Log

(append entries below this line)

## [2026-04-24 17:30] Stage 0 / T0.1 — Inference engine = vLLM 0.19.1 (not 0.20.0)

**Decision:** Pin `vllm==0.19.1` as the project's inference engine. Python 3.12 (resolved to 3.12.0 via pyenv).

**Alternatives considered:**
- vLLM 0.20.0 — exists as a git tag and GitHub release (2026-04-23), but marked `PRERELEASE: True`; not on PyPI `info.version`. Would need `uv add vllm==0.20.0rc1` or similar and risk pre-GA bugs.
- SGLang 0.5.10 — viable fallback; less verified for Qwen 3.6 and Qwen 3 thinking-mode toggles per the Stage 0 exploration.

**Reason:** vLLM 0.19.1 is the current stable PyPI release (uploaded 2026-04-18). Its registry already includes `Qwen3_5ForConditionalGeneration` (covers Qwen 3.6-27B) and `Gemma4ForConditionalGeneration`, and its predecessor 0.19.0 added Blackwell sm_120 CUTLASS blockwise FP8 GEMM (release notes cite PR #37970) — matches our 5090 hardware. Transformers v5 support landed in 0.19.1. Going stable > prerelease for the first locked env.

**Source:**
- https://pypi.org/pypi/vllm/json — `info.version == "0.19.1"` uploaded 2026-04-18T05:49:16.
- https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.19.0 — Blackwell SM120 fp8 GEMM (#37970), Gemma 4 architecture support (#38826), Transformers v5 adopted.
- https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.19.1 — 10+ Gemma 4 bug fixes; transformers v5.5.3 pin.
- https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/vllm/model_executor/models/registry.py — registry confirms Qwen3_5 + Gemma4 arch classes at 0.19.1 tag.

**Reversibility:** medium. Flipping to 0.20.0rc / 0.18.x / SGLang requires re-running `uv lock` + re-validating model load tests.

**How to revert:** bump `vllm==` in `pyproject.toml`, `uv lock && uv sync`, re-run `scripts/smoke_load.py`.

**Downstream dependencies:** every Stage 0/1/2+ model load. Affects CONVENTIONS.md "Inference engine" and "Python version" entries; affects Stage 2 T2.3 capper + T2.4 judge driver API surface.

---

## [2026-04-24 17:35] Stage 0 / T0.1 — Torch version resolved to 2.10.0+cu128 (not 2.11.0+cu130)

**Decision:** Accept `torch==2.10.0+cu128` as pulled transitively by vllm 0.19.1 (not 2.11.0+cu130 as the initial research suggested for vLLM 0.20.0).

**Alternatives considered:**
- Force `torch==2.11.0+cu130` via explicit pin and custom index URL — risks breaking vLLM 0.19.1's own pin.

**Reason:** vLLM 0.19.1's wheel spec pulls torch 2.10.0+cu128. Our driver 580.126.09 advertises CUDA 13.0 max runtime which is backward-compatible with cu128. `torch.cuda.is_available()` works, both 5090s report sm_120 compute capability, 33.7 GB each. No need to fight the resolver.

**Source:**
- `uv sync` output: `+ torch==2.10.0+cu128`.
- `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"` → `2.10.0+cu128 12.8`.
- `nvidia-smi` driver=580.126.09, CUDA Version advertised 13.0.

**Reversibility:** low-medium. Would need overriding vLLM's torch pin via `tool.uv.override-dependencies`, which might break other pkgs.

**How to revert:** only if a later blocker demands it; add override in pyproject and re-test everything.

**Downstream dependencies:** none directly — our code doesn't rely on torch 2.11 features.

---

## [2026-04-24 17:40] Stage 0 / T0.4 — Gemma 2 27B FP8 checkpoint = `Infermatic/gemma-2-27b-it-FP8-Dynamic`

**Decision:** Use `Infermatic/gemma-2-27b-it-FP8-Dynamic` as the quantized Gemma 2 27B subject checkpoint.

**Alternatives considered:**
- `nm-testing/gemma-2-27b-it-FP8` and `neuralmagic/gemma-2-27b-it-FP8` — the Stage 0 research report named these, but HF API returns 401 on both (repos don't exist — HF's 401-for-unknown-repo behavior when unauthenticated; authenticated `HfApi().model_info(...)` would 404).
- `dangvansam/gemma-2-27b-it-FP8-fix-system-role` (17 dls) — niche fork that tweaks system-role handling; unnecessary complication.
- `mbley/google-gemma-2-27b-it-AWQ` (454 dls) — AWQ fallback if FP8 fails.

**Reason:** `Infermatic/gemma-2-27b-it-FP8-Dynamic` has the highest community download count (134) of the existing Gemma 2 27B FP8 variants on HF. Config confirms `Gemma2ForCausalLM` / 46 layers / 4608 hidden / `quant_method: fp8`. FP8-Dynamic (activations quantized at runtime) is supported by vLLM. No official Google FP8 for Gemma 2 27B exists, so a community variant is required; this is the most-battle-tested option.

**Source:**
- https://huggingface.co/api/models?search=gemma-2-27b-it+fp8 — searched and ranked by downloads, 2026-04-24.
- https://huggingface.co/Infermatic/gemma-2-27b-it-FP8-Dynamic/raw/main/config.json — config verified, `architectures=['Gemma2ForCausalLM']`, `quantization_config.quant_method='fp8'`.

**Reversibility:** high. If extraction-fidelity check fails in Stage 3 T3.1.0, switch to the AWQ fallback `mbley/google-gemma-2-27b-it-AWQ` and re-run.

**How to revert:** change the model ID in `configs/subjects.yaml` (to be created in Stage 2 T2.1) and re-run the quant-validity check.

**Downstream dependencies:** Stage 3 T3.1.0 quant-validity check; all Stage 3/4/6 experiments that use Gemma 2 27B.

---

## [2026-04-24 18:10] Stage 0 / T0.7 — Paper artifacts audit: HF dataset has more than expected

**Decision:** Use the full `lu-christina/assistant-axis-vectors` HF dataset (not just AA directions) as input for Stage 3 Tier 1 PCA. Skip rollout regeneration for Tier 1 PCA fit; still regenerate rollouts for the τ-calibration distribution needed by Stage 4 T4.0.

**Alternatives considered:**
- Regenerate Tier 1 role + trait vectors from scratch to ensure provenance consistency — wastes ~1 day of GPU for no scientific gain; paper's pre-computed vectors were made on bf16 which is the reference we're validating our fp8 extraction against.

**Reason:** The HF dataset (1.2 GB cached to `data/paper_artifacts/assistant_axis_vectors/`) contains per-subject (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B):
  - `assistant_axis.pt` — shape `[n_layers, d_model]` bf16; AA direction at every layer.
  - `default_vector.pt` — mean default-Assistant activation.
  - `role_vectors/<role>.pt` — 275 per subject.
  - `trait_vectors/<trait>.pt` — 240 per subject.
  - `capping_config.pt` — per-layer contrast vectors `contrast_role_pos3_default1` (Qwen + Llama only; **Gemma 2 27B is missing this file — confirmed**).
  - Raw rollouts and τ-calibration distributions are **NOT** released.

**Implications:**
- Stage 3 T3.1 Tier 1 PCA: reads role/trait vectors directly, skips generation.
- Stage 3 T3.1.0 quant-validity check for Tier 1 (Gemma 2 27B, Qwen 3 32B): projects our-quantized-model's test-prompt activations onto the paper's bf16 AA direction. Ready to use.
- Stage 4 T4.0 capping for Gemma 2 27B: must transcribe layer range from paper Appendix F (no capping_config.pt); Qwen 3 32B reads capping_config.pt OR uses paper line 691 transcription directly (`configs/paper_capping_ranges.yaml`).
- **τ-calibration still requires rollout regeneration** for all subjects, since paper didn't release per-rollout projection distributions. This is a Stage 3 T3.1 cost for all 4 subjects (not just Tier 2).

**Source:**
- `huggingface-cli download --repo-type dataset lu-christina/assistant-axis-vectors --local-dir data/paper_artifacts/assistant_axis_vectors` — 1.2 GB pulled 2026-04-24.
- `external/assistant-axis` pinned at commit `a989619`, message "Update jailbreak_capped.json".
- File inspection via `torch.load(...)` — schemas documented above.

**Reversibility:** high. If we decide to regenerate Tier 1 from scratch, wipe `data/paper_artifacts/assistant_axis_vectors/`, run the extraction pipeline, done.

**How to revert:** set a `configs/extraction.yaml` flag `use_paper_tier1_vectors: false`; re-run Stage 3 T3.1.

**Downstream dependencies:** Stage 3 T3.1 (skips Tier 1 PCA input gen), T3.1.0 (uses paper's bf16 AA for fidelity check), T3.5 (role-vector PCA reads from paper or regenerated cache). Stage 4 T4.0 capping-layer-range config.

