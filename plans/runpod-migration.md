# Runpod Migration Contingency

**Status as of 2026-04-25:** **Not active.** The live execution path is local **4× RTX 5090, bf16, TP=4**. This document exists so a future agent can switch the project to Runpod cloud GPUs cleanly when the local cluster becomes unavailable.

**Trigger to use this doc:** the local 4× RTX 5090 cluster is reclaimed by another workload, or scheduling pressure forces a cloud-only path, or the user explicitly says "let's move to Runpod."

**Authoring context:** drafted in conversation with the user covering throughput benchmarks, prod-pod safety, lifecycle/SDK choices, volume strategy, and file-level diff inventory. The user has the Runpod CLI installed and the Python SDK key configured. They have **one prod pod that must never be touched.**

---

## 🚨 CRITICAL — Prod pod isolation

**Prod pod ID: `u3qiwwpdxqaypp`. NEVER stop, never destroy, never modify, never list as ours.**

This pod runs unrelated production work on the user's account. Three layers of defense MUST be implemented before any other Runpod work:

**Layer 1 — config:**
- Add `RUNPOD_PROD_POD_ID=u3qiwwpdxqaypp` to `.env` (gitignored).
- Add it to `.env.example` (committed) without the value.
- Document in `CONVENTIONS.md` under a new "Runpod infrastructure" section.

**Layer 2 — SDK wrapper guard** (`src/runpod/lifecycle.py`):
```python
def destroy_pod(pod_id: str) -> None:
    if pod_id == os.environ["RUNPOD_PROD_POD_ID"]:
        raise RuntimeError(f"refusing to destroy prod pod {pod_id}")
    if not _pod_has_project_tag(pod_id):
        raise RuntimeError(f"pod {pod_id} not tagged as project — refusing to destroy")
    runpod.api.delete_pod(pod_id)  # SDK call
```
NO raw `runpod.api.delete_pod` or `runpodctl remove pod` anywhere outside this wrapper. Add a `ruff` rule or pre-commit grep check that fails CI if either appears in `src/`, `scripts/`, or `tests/` outside `src/runpod/lifecycle.py`.

**Layer 3 — listing filter** (`src/runpod/lifecycle.py::list_pods`):
- Filter pods by tag `project=multi-axis-persona-safety`.
- Prod pod has no such tag → never appears in our listings → cannot be passed to destroy by accident.

---

## Locked decisions from prior discussion (do not re-litigate)

These are user-confirmed. Apply them as-is when migrating.

| # | Decision | Rationale |
|---|----------|-----------|
| RP-1 | **Pod lifecycle = destroy, not stop.** | Stopped Runpod pods can have their GPUs reclaimed; storage charges accrue; no upside vs destroy. Volume holds all state. |
| RP-2 | **Use the Python SDK (`runpod` on PyPI), not `runpodctl` shell-out.** | Typed objects, structured errors, async support. The `runpodctl` CLI skill stays for Claude's interactive ops during dev only. |
| RP-3 | **GPU = RTX Pro 6000 Blackwell, 96 GB GDDR7 VRAM** (NOT 188 GB — that was system RAM confusion). $1.70/hr at the time of discussion. | All 4 core subjects fit bf16 on a single 96 GB GPU with KV headroom. Llama 3.3 70B is the only subject that requires FP8 (~70 GB) at 96 GB; bf16 70B (140 GB) needs 2× Pro 6000. |
| RP-4 | **Single-GPU pod by default (TP=1, no tensor parallelism).** Spin a 2× GPU pod only for compute-bound stretches if throughput economics favor it. | Simplifies vLLM config and the subprocess wrapper from Stage 2 T2.1.6. |
| RP-5 | **Network volume = 1 TB, single shared volume mounted at `/workspace`.** | Holds HF model cache + activation caches + results. Same volume attaches to every pod we spawn. |
| RP-6 | **Pods must be tagged `project=multi-axis-persona-safety` at creation time.** | Required by the Layer-3 listing filter; no tag = invisible to our wrappers = unmanageable by us = safe. |
| RP-7 | **Phased topology stays unchanged.** Subject pod → tear down → judge pod → tear down → cross-check pod → tear down. Each phase = single-GPU pod, single model. Volume persists between. | Same as the current local plan. |
| RP-8 | **Pod-destroy IS the VRAM cleanup mechanism on Runpod.** The Stage 2 T2.1.6 subprocess wrapper exists to fight vLLM TP-tear-down VRAM leaks on local hardware. On Runpod, destroying the pod is the strongest possible cleanup — the subprocess wrapper becomes redundant for cleanup but remains useful as a code-isolation/serialization boundary. Keep the wrapper; document the redundancy. | |

---

## Open decisions the migrating agent must surface to the user

Do NOT pick these silently. Surface them, get user sign-off, log to `decisions.md`.

### OD-1 — Budget ceiling

**Status:** unresolved. The user's original $150 ceiling was discussed before the dual-dataset (DAN + Shah-reconstructed) decision doubled the safety-eval compute. After dual-dataset and the bf16/TP=4 decisions, my latest estimate puts core stages at **~$215-340** at $1.70/hr.

Three options were tabled but never picked:
- **Option X — raise budget to $250** (with the new estimate, probably needs $300-400 to be safe). No scope cuts.
- **Option Y — keep $150, take both cuts** (skip Tier 1 rollout regen + defer Gemma 4 31B thinking ON to Ext 2). Now infeasible because Stage 0 T0.7 audit (decisions.md 2026-04-24 18:10) confirmed HF *does* have role vectors but *not* rollout distributions, so Tier 1 rollout regen is partly unavoidable for τ-calibration.
- **Option Z — keep $150, run Stage 0 T0.7 audit first, then decide.** The audit happened. Net savings: ~$20-30 for skipping Tier 1 PCA-fit rollouts; τ-calibration regen still needed.

**Recommended: ask the user to pick X (with the revised ~$300 estimate) or X+Y combined cuts to fit ~$200.**

### OD-2 — Datacenter region

The volume must be in the same Runpod datacenter as the GPUs we rent. RTX Pro 6000 Blackwell availability varies by DC (commonly EU-RO-1, US-CA-2 — verify via `runpod.api.get_gpu_types()` or the Runpod console). User must pick the DC before creating the volume.

### OD-3 — Llama 3.3 70B in core or in Ext 9?

Currently deferred to Ext 9 because bf16 70B (140 GB) doesn't fit the local 4× 5090 budget (128 GB total). On Runpod:
- 1× Pro 6000 (96 GB): Llama 70B at **FP8 (~70 GB) fits**. Could promote back to core.
- 2× Pro 6000 (192 GB): Llama 70B at **bf16 fits**. Could promote at higher hourly cost.

Ask the user: do you want Llama 70B back in core now that bf16 isn't blocked? If yes, FP8 (paper itself anticipates fp8/int8 on constrained hardware — Stage 0 T0.4 in original plan) is the cheapest path.

### OD-4 — TP=1 single-GPU vs TP=2 dual-GPU pod

Single-GPU is cheaper per hour ($1.70 vs $3.40) and simpler. Dual-GPU gives ~1.6-1.8× throughput on 27-32B models in batched inference. For our token-bound workload, single-GPU at $1.70 is usually better $/throughput. **Default to single-GPU; let the user upgrade to dual if Stage 4 sweeps run too slow.**

### OD-5 — Cost-alert threshold

Recommended: alert at 80% of the agreed ceiling. If $250: alert at $200. If $300: alert at $240. Surface in `src/runpod/cost.py` with a hard `raise` at 100%.

---

## What changes from local 4× 5090 → Runpod single Pro 6000

The science is unchanged. The infra and a few precision/parallelism settings are not.

### Hardware & precision (CONVENTIONS.md, CLAUDE.md)

| Aspect | Local 4× 5090 (current) | Runpod 1× Pro 6000 |
|---|---|---|
| GPUs | 4× RTX 5090, 32 GB each, 128 GB total | 1× RTX Pro 6000 Blackwell, 96 GB |
| Tensor parallel | TP=4 | TP=1 (or TP=2 if dual-GPU pod) |
| Precision | bf16 across all subjects + judges | bf16 for 27-32B subjects + judges; FP8 for Llama 70B if promoted to core (fits in 96 GB) |
| Subjects in core | Gemma 2 27B, Qwen 3 32B, Gemma 4 31B (× 2 modes) — Llama 70B in Ext 9 | Same 4 by default; Llama 70B optionally promoted (see OD-3) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | `0` (single GPU) or `0,1` (dual) |
| VRAM cleanup | Subprocess wrapper (Stage 2 T2.1.6) — vLLM TP-tear-down leaks ~25 GB/GPU + 6 semaphores | Pod destroy = nuclear cleanup. Wrapper kept for code isolation but cleanup is redundant. |

### Data & filesystem (CONVENTIONS.md, every stage plan that writes paths)

| Path (local) | Path (Runpod) |
|---|---|
| `data/cache/activations/<subject>/...` | `/workspace/cache/activations/<subject>/...` |
| `data/cache/lmsys_norms/...` | `/workspace/cache/lmsys_norms/...` |
| `data/cache/assistant_axis/...` | `/workspace/cache/assistant_axis/...` |
| `data/paper_artifacts/...` | `/workspace/paper_artifacts/...` |
| `data/eval/...` | `/workspace/eval/...` |
| `results/exp{N}_*/...` | `/workspace/results/exp{N}_*/...` |
| HF model cache (default `~/.cache/huggingface`) | `/workspace/hf_cache` (set `HF_HOME=/workspace/hf_cache`) |

Use a config-driven `DATA_ROOT` env var so the same code runs locally (DATA_ROOT=`./data`) and on Runpod (DATA_ROOT=`/workspace`). Add to `src/utils/env.py`.

### Lifecycle & cost (new)

- **New module `src/runpod/`** with `lifecycle.py`, `sync.py`, `cost.py` (see "File-by-file edits" below).
- **Pod naming:** `map-{stage}-{task}-{YYYYMMDD-HHMM}` (e.g., `map-stage3-extract-20260501-1430`). Tag `project=multi-axis-persona-safety` always.
- **Pod creation:** every spawn attaches the volume at `/workspace` and sets env vars `HF_HOME=/workspace/hf_cache`, `DATA_ROOT=/workspace`, `HF_TOKEN=<from-secret>`, `HF_HUB_ENABLE_HF_TRANSFER=1`, plus the project's `seed`.
- **Cost ledger:** `src/runpod/cost.py` polls each pod's uptime via SDK at destroy time, multiplies by GPU hourly rate, appends to `/workspace/results/cost_ledger.parquet` with `(pod_id, stage, task, started_at, destroyed_at, gpu_type, gpu_count, hours, dollars, note)`.

### What does NOT change

- Hypotheses H1-H4, experiments, statistical framework.
- Paper-derived conventions: AA as primary intervention direction, layer-scope rules (single-layer steering, multi-layer capping via 2D sweep), τ-calibration distribution definition, role-vector fully/somewhat split, judge prompts, binarization rule, dual-dataset rule (DAN + Shah-reconstructed).
- Stage 1 architecture (`src/extraction`, `src/steering`, `src/evaluation`, `src/analysis`, `src/visualization`, `src/utils`).
- Subprocess wrapper (Stage 2 T2.1.6) — kept as code-isolation boundary; cleanup function becomes redundant.
- Decision-logging protocol (`decisions.md`).
- Phased topology (subject → judge → cross-check, each phase a separate model load).

---

## File-by-file edit inventory

Apply in roughly this order:

### 1. `CLAUDE.md` — Hardware, Precision policy, Current State

Replace the "Hardware" line under Models:
> ~~Hardware: 4× RTX 5090 available (32 GB each = 128 GB total). All 4 GPUs accessible to this project as of 2026-04-25...~~

→

> Hardware: **Runpod cloud, 1× RTX Pro 6000 Blackwell (96 GB) per pod by default; single-GPU TP=1.** Pods are ephemeral; persistent state on a 1 TB network volume mounted at `/workspace`. Prod pod `u3qiwwpdxqaypp` exists on the same account and is **never touched**. See `plans/runpod-migration.md` for the lifecycle, safety, and SDK conventions; see `src/runpod/lifecycle.py` for the wrapper.

Replace the "Precision policy" subsection's TP=4 framing with TP=1. Bf16 stays for 27-32B subjects + judges. Add a note about FP8 for Llama 70B if it's promoted to core (per OD-3).

Update "Current State" — add a line: `Runpod migration applied <DATE>. See plans/runpod-migration.md and decisions.md entry "Runpod migration".`

### 2. `plans/CONVENTIONS.md` — new "Runpod infrastructure" section

Add under Locked decisions, after "Inference & serving":

```markdown
### Runpod infrastructure

- **Cloud provider:** Runpod. SDK = `runpod` PyPI package (NOT shell-out to `runpodctl`).
- **Default pod spec:** 1× RTX Pro 6000 Blackwell, 96 GB VRAM, TP=1, bf16 precision (FP8 reserved for Llama 70B if used).
- **Network volume:** 1 TB, single shared, mounted at `/workspace`. Volume ID stored in `.env` as `RUNPOD_VOLUME_ID`. Datacenter pinned at volume creation; all subsequent pods MUST be in the same DC.
- **Filesystem layout on volume:** `/workspace/{cache, eval, paper_artifacts, hf_cache, results}/`. Code resolves via `DATA_ROOT` env var (default: `./data` locally, `/workspace` on Runpod).
- **Pod naming:** `map-{stage}-{task}-{YYYYMMDD-HHMM}`. Every pod tagged `project=multi-axis-persona-safety`.
- **Pod lifecycle:** destroy after each phase, never stop. `src/runpod/lifecycle.py` is the only module allowed to call destructive SDK methods. Raw `runpod.api.delete_*` outside that module is banned (enforced by pre-commit grep).
- **Prod pod safety:** `RUNPOD_PROD_POD_ID=u3qiwwpdxqaypp` in `.env`. The wrapper refuses any destructive op against this ID and against any pod missing the project tag.
- **HF model cache:** `HF_HOME=/workspace/hf_cache`, set on every pod at spawn. First load populates; subsequent pods hit volume cache for free.
- **Cost ledger:** `/workspace/results/cost_ledger.parquet`, schema `(pod_id, stage, task, started_at, destroyed_at, gpu_type, gpu_count, hours, dollars, note)`. Cost-alert threshold via `RUNPOD_COST_ALERT_USD` env var; hard stop via `RUNPOD_COST_CEILING_USD`.
```

Also update "Inference & serving" — the phased topology bullet stays, but reword "subject model on all 4 GPUs" → "subject model on the pod's GPUs".

### 3. `plans/decisions.md` — append migration entry

Use the template at the top of the file. Required fields: Decision (migrated to Runpod), Alternatives considered (stay local with reduced scope, GCP/AWS), Reason (local cluster reclaimed / explicit user request), Source (this migration doc + user instruction), Reversibility (medium — code paths swap via env var, but volume data has to stay accessible), How to revert (swap `DATA_ROOT` back to `./data`, restore TP=4 precision settings; volume data persists indefinitely on Runpod), Downstream dependencies (every stage from Stage 0 onward).

### 4. `plans/stage-0-environment.md` — major overhaul

**New T0.0 (insert at the top of the Tasks section): Runpod infrastructure setup.**
Sub-steps:
- Confirm `runpod` Python SDK installed; verify auth via `runpod.api.get_gpu_types()`.
- Create the network volume (1 TB) in chosen DC. Save the volume ID to `.env` as `RUNPOD_VOLUME_ID`.
- Pull `RUNPOD_PROD_POD_ID=u3qiwwpdxqaypp` from user, write to `.env`.
- Implement `src/runpod/lifecycle.py` with `spawn_pod`, `wait_ready`, `ssh_exec`, `destroy_pod`, `list_pods` (all guarded — see Layer 1/2/3 above).
- Add pre-commit grep that fails on raw `runpod.api.delete_*` or `runpodctl remove` outside `src/runpod/lifecycle.py`.
- Spawn a 1-GPU smoke pod, SSH in, run `nvidia-smi`, verify volume mounted at `/workspace`, destroy. Verify cost ledger entry written.

**T0.1 (inference engine):** unchanged decision (vLLM 0.19.1) but verify it loads on Pro 6000 — sm_120 wheels confirmed in decisions.md 2026-04-24 17:30. No change expected.

**T0.3 (env install):** rewrite as "build provisioning script that runs on pod boot." Either:
- Bake a Docker image with the env (fastest cold-start, harder to iterate on).
- Boot script: clone repo → `uv sync` → mount volume → ready (slower cold-start by ~3-5 min, easy to update).
- Recommend boot script for project velocity; can switch to image later.

**T0.4 / T0.5 (model loading):** add "first pod that loads each model populates `/workspace/hf_cache`; subsequent pods hit cache. Verify cache hit on second spawn." Bf16 27-32B fits single 96 GB. Llama 70B FP8 (if promoted) fits at ~70 GB.

**T0.7 (paper artifacts):** unchanged content; just write to `/workspace/paper_artifacts/` instead of `data/paper_artifacts/`. Decisions.md 2026-04-24 18:10 already documents what was pulled.

**T0.8 / T0.9 (judge setup):** unchanged content; reframe phased topology as "spawn judge pod → classify → destroy" instead of "load judge on all 4 GPUs → classify → tear down."

**T0.10 (eval datasets):** download to `/workspace/eval/`. The DAN dataset and Shah-reconstructed datasets are already locked per CONVENTIONS "Jailbreak datasets" — just write to volume.

**T0.11 (baseline GPU util test):** rerun on Pro 6000. Target throughput numbers: see "Compute & cost reference data" below. Acceptance criteria adjusted from ≥90% util at TP=4 to ≥85% util at TP=1 (single-GPU batched is harder to saturate on Blackwell with 96 GB headroom).

**T0.12 (handoff):** unchanged.

### 5. `plans/stage-1-architecture.md` — minor

Add `src/runpod/` to the directory structure. Add `DATA_ROOT` env-var resolution to `src/utils/env.py` (T1.1 or wherever env utilities live).

### 6. `plans/stage-2-infrastructure.md` — surgical

The Stage 2 plan currently locks `bf16/TP=4` with a CRITICAL tear-down rule citing 4× 5090-specific VRAM leaks (decisions.md 2026-04-25 fp8→bf16 entry). Adjustments:

- **Operating regime block:** swap "4× RTX 5090 (128 GB total)... TP=4" → "1× RTX Pro 6000 Blackwell (96 GB)... TP=1". Note that the subprocess wrapper from T2.1.6 stays for code isolation, but the VRAM-leak cleanup motivation is gone (pod-destroy is the cleanup).
- **D3 (HF backend on bf16 27-32B models):** `device_map="auto"` collapses to single-GPU placement. Bf16 27B (~54 GB), 31B (~62 GB), 32B (~64 GB) all fit in 96 GB with KV headroom. Confirm with a smoke load.
- **D6 (judge runtime probe):** acceptance criteria stays ≥30 labels/sec; values may shift on Pro 6000 vs 4× 5090 — re-measure.
- **D11 (smoke test):** change "Gemma 2 27B at bf16/TP=4" → "Gemma 2 27B at bf16/TP=1". Wall-clock budget stays ≤2 hours.
- **T2.1.6 (subprocess wrapper):** keep the wrapper API; remove the leak-cleanup justification, replace with "process-isolation guarantee for sequential model loads on the same pod."
- **T2.4 (orchestrator):** between phases, the orchestrator now calls `lifecycle.destroy_pod` and `lifecycle.spawn_pod` for the next phase. Phase parquet writes go to `/workspace/results/...` so the next pod reads them via mounted volume.

### 7. `plans/stage-3-foundation.md`, `stage-4-attack.md`, `stage-5-composition.md`, `stage-6-defense.md`, `stage-7-extensions.md` — path swaps

Search-and-replace `data/cache/` → `/workspace/cache/` and `results/` → `/workspace/results/`. Or — preferred — make every path read from `DATA_ROOT` so a single env var flips local↔Runpod.

Each long-running task adds a one-line cost note: estimated wall-clock on Pro 6000 + dollars at $1.70/hr.

Stage 4 T4.0's 2D capping sweep is the largest budget block (estimated $80-130 with dual-dataset). Surface this estimate explicitly in the task description.

### 8. `plans/stage-8-presentation.md` — minor

Final results pull: rsync from `/workspace/results/` to laptop for plotting/writing. Add a "post-experiment" sub-task: "Pull final figures and `details.parquet` artifacts to laptop; archive volume snapshot to HF dataset for reproducibility."

### 9. `plans/plan.md` — agent onboarding update

Add to the onboarding checklist:
> "If you see `RUNPOD_VOLUME_ID` set in `.env`, the project is on Runpod. Read `plans/runpod-migration.md` Section "Locked decisions" before any pod operation. **Never call destructive Runpod operations outside `src/runpod/lifecycle.py`. Never touch pod `u3qiwwpdxqaypp`.**"

### 10. New code: `src/runpod/`

Three modules. Skeleton:

```python
# src/runpod/lifecycle.py
import os
import runpod  # Python SDK
from runpod.api import create_pod, get_pod, delete_pod, get_pods

PROJECT_TAG = "project=multi-axis-persona-safety"
DEFAULT_GPU_TYPE = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"  # confirm exact SKU name via runpod.api.get_gpu_types()

def spawn_pod(stage: str, task: str, gpu_count: int = 1, image: str = "<our-image-or-template>") -> dict:
    """Spawn a project-tagged pod with our volume attached. Returns pod metadata."""
    name = f"map-{stage}-{task}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    env = {
        "HF_HOME": "/workspace/hf_cache",
        "DATA_ROOT": "/workspace",
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONHASHSEED": str(os.environ.get("SEED", "42")),
    }
    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=DEFAULT_GPU_TYPE,
        gpu_count=gpu_count,
        volume_id=os.environ["RUNPOD_VOLUME_ID"],
        volume_mount_path="/workspace",
        env=env,
        # Tag fields differ per SDK version — confirm; fallback: name suffix encodes tag
    )
    return pod

def destroy_pod(pod_id: str) -> None:
    if pod_id == os.environ["RUNPOD_PROD_POD_ID"]:
        raise RuntimeError(f"refusing to destroy prod pod {pod_id}")
    if not _pod_has_project_tag(pod_id):
        raise RuntimeError(f"pod {pod_id} missing project tag — refusing to destroy")
    runpod.api.delete_pod(pod_id)

def list_pods() -> list[dict]:
    """Returns ONLY project-tagged pods. Prod pod is invisible by construction."""
    all_pods = runpod.api.get_pods()
    return [p for p in all_pods if _pod_has_project_tag(p["id"])]

def _pod_has_project_tag(pod_id: str) -> bool:
    pod = runpod.api.get_pod(pod_id)
    # Tag mechanism depends on SDK version: name prefix `map-` is the fallback
    return pod.get("name", "").startswith("map-")

def wait_ready(pod_id: str, timeout_s: int = 600) -> None:
    """Poll pod status until SSH-ready or timeout."""
    ...

def ssh_exec(pod_id: str, command: str) -> str:
    """SSH into pod, run command, return stdout."""
    ...
```

```python
# src/runpod/sync.py
def push_code(pod_id: str) -> None:
    """SSH into pod, git pull or rsync the project repo."""
    ...

def pull_results(pod_id: str, dest: Path) -> None:
    """Rsync /workspace/results/ from pod to local dest."""
    ...
```

```python
# src/runpod/cost.py
def log_pod_cost(pod_id: str, stage: str, task: str, note: str = "") -> None:
    """At destroy time, compute uptime × hourly rate, append row to /workspace/results/cost_ledger.parquet."""
    ...

def total_spend() -> float:
    """Sum cost_ledger; raise if exceeds RUNPOD_COST_CEILING_USD."""
    ...

def alert_if_above_threshold() -> None:
    """Soft alert at RUNPOD_COST_ALERT_USD."""
    ...
```

Tests: at least one integration test that spawns + destroys a CPU-only pod (no GPU charge) to verify the wrapper end-to-end, plus unit tests for prod-pod-guard and tag-filter logic.

### 11. `pyproject.toml`

Add `runpod` to dependencies. Confirm version pin via `uv add runpod` and run lockfile.

---

## Compute & cost reference data

### Throughput benchmarks pulled during prior discussion

(Sources at the bottom; numbers from databasemart, Spheron, Pulsed Media, NVIDIA InferenceMAX. All for batched vLLM unless noted.)

| Model | Precision | Concurrency | tok/s |
|---|---|---|---|
| Qwen-32B | FP16 | 50 | 829 |
| Qwen-32B | FP16 | 300 | 1,654 |
| Llama 3.3 70B | INT4 AWQ | high (100+) | 8,425 |
| Llama 3.3 70B | Q4_K_M (llama.cpp) | 1 (single-stream) | 34 |

Estimates (interpolation; verify on T0.11):
- Gemma 2 27B bf16 batched: ~1,000 tok/s aggregate at 50 concurrent
- Gemma 4 31B bf16 batched: ~900 tok/s aggregate at 50 concurrent
- Llama 3.3 70B FP8 batched (if promoted): ~600-1,500 tok/s at 50 concurrent — wide bracket pending T0.11 measurement

### Wall-clock + cost estimates (single-GPU Pro 6000 at $1.70/hr, dual-dataset rule applied)

These are upper bounds. Actuals likely 70-85% of these.

| Stage / work unit | Wall-clock (hrs) | Cost ($) |
|---|---|---|
| Stage 0 (env + smoke + paper artifacts) | 4-6 | $7-10 |
| Tier 1 rollout regen for τ-calibration (3 subjects × 82K rollouts each) | 22-28 | $37-48 |
| Tier 2 extraction (Gemma 4 31B × 2 modes × 82K rollouts) | 14-18 | $24-31 |
| Baseline safety (DAN + Shah, dual) + capability per subject (×4) | 16-24 | $27-41 |
| Stage 4 steering sweeps + 2D AA-capping calibration (DUAL DATASET — 2× safety eval) | 50-80 | $85-136 |
| Stage 5 composition (~50 pairs × 200 rollouts × 4 subjects) | 4-6 | $7-10 |
| Stage 6 defense (Phase A + conditional B, dual dataset) | 16-25 | $27-43 |
| Judge classification across all experiments (Qwen 3.6-27B + Gemma 4 31B-it cross-check) | 7-10 | $12-17 |
| Buffer for reruns / debug / pod spin-up overhead (~10%) | 13-20 | $22-34 |
| **Total (4 core subjects, no Llama 70B)** | **~146-217 hrs** | **~$248-370** |

**If Llama 70B promoted to core (FP8, 3 Tier 1 subjects):** add ~30-40 hrs for rollouts + steering across Llama → +$50-70.

**Volume cost:** ~$0.07/GB/month × 1 TB ≈ **$70/month** while volume exists. Budget 1-2 months → $70-140 separate from compute.

### How this compares to the user's last-discussed $150 ceiling

Current estimate ($248-370 compute + $70-140 volume) is **2-3× the original $150 ceiling.** The migrating agent must surface this gap and get user sign-off on a new ceiling before making code edits.

If $150 is hard, scope cuts available:
- Drop Shah-reconstructed dataset, keep only DAN (halves Stage 4 + Stage 6 safety eval): saves ~$60-90.
- Defer Gemma 4 31B thinking ON to Ext 2 (cuts 1 of 4 subjects in Stages 3-6): saves ~$40-60.
- Cut Stage 4 steering strengths from 9 to 5 (already cut for capability eval; extend to safety): saves ~$30-50.
- Skip Stage 4 T4.0 Tier 2 2D sweep, use Tier 1 capping ranges as proxy for Tier 2: saves ~$25-40.

Combined cuts could land $130-180. Still tight.

---

## Implementation sequence (ordered checklist for the migrating agent)

Approximate sequencing. Each entry should produce a `decisions.md` log entry where it picks a non-obvious value.

1. **Surface OD-1 through OD-5 to the user.** Get sign-off on budget, DC, Llama 70B inclusion, GPU count, alert threshold. Log to `decisions.md`.
2. **Add `runpod` SDK + `RUNPOD_*` env vars to `.env.example`.** Verify auth: `python -c "import runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; print(runpod.api.get_gpu_types()[:5])"`.
3. **Implement `src/runpod/lifecycle.py`** with the three guards (prod-pod ID, project tag, no-raw-SDK). Add unit tests for the guards.
4. **Implement pre-commit grep check** to enforce no raw destructive SDK calls outside the wrapper.
5. **Create the network volume** (1 TB, chosen DC). Record `RUNPOD_VOLUME_ID` to `.env`. Don't forget to add it to `.env.example` without the value.
6. **Smoke test the wrapper:** spawn CPU-only pod (cheapest), SSH in, mount-check `/workspace`, destroy. Verify cost ledger entry.
7. **Add `DATA_ROOT` env-var resolution to `src/utils/env.py`.** Default to `./data` locally, `/workspace` on Runpod. Have all `data/...` and `results/...` paths in the codebase resolve through this.
8. **Edit `CLAUDE.md`** Hardware + Precision sections + Current State.
9. **Edit `plans/CONVENTIONS.md`** — add Runpod infrastructure section; update Inference & serving to drop "all 4 GPUs" framing.
10. **Edit `plans/stage-0-environment.md`** — insert new T0.0 (Runpod lifecycle setup), update T0.3, T0.4, T0.5, T0.7-T0.11 as listed above.
11. **Edit `plans/stage-1-architecture.md`** — add `src/runpod/` to directory tree, add DATA_ROOT to env utilities.
12. **Edit `plans/stage-2-infrastructure.md`** — flip operating regime block to TP=1, adjust D3, D6, D11; rewrite T2.1.6 motivation; update T2.4 orchestrator.
13. **Edit `plans/stage-3-foundation.md` through `plans/stage-7-extensions.md`** — path swaps via DATA_ROOT (most edits are mechanical).
14. **Edit `plans/stage-8-presentation.md`** — add results-pull task.
15. **Edit `plans/plan.md`** — agent onboarding adds Runpod safety reminder.
16. **Append `decisions.md` entry** "Runpod migration applied <DATE>" with full decision/alternatives/reason/source/reversibility/revert/dependencies.
17. **Run a bounded smoke test on Runpod:** Stage 2 T2.9 smoke (Gemma 2 27B, 100 prompts, 50 capability problems each, AA-cap). Wall-clock budget ≤2 hours; cost budget ≤$5. Confirm phased pipeline works on cloud.
18. **If smoke passes, hand off to user with a fresh "Stage 2 → Stage 3 Handoff" block in `progress.md` that names the live Runpod pod (if any), the volume ID, and the cost-to-date snapshot.**

---

## Sources

- Pro 6000 vLLM Inference Benchmark — https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000
- RTX PRO 6000 Benchmarks (30B AWQ, 70B FP8) — https://www.spheron.network/blog/rent-nvidia-rtx-pro-6000/
- RTX PRO 6000 Blackwell for LLMs (96GB) — https://vrlatech.com/rtx-pro-6000-blackwell-for-llms-why-96gb-changes-everything/
- NVIDIA RTX Pro 6000 Blackwell — https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
- NVIDIA RTX Pro 6000 wiki (Pulsed Media) — https://wiki.pulsedmedia.com/wiki/NVIDIA_RTX_Pro_6000_(Blackwell)
- RTX PRO 6000 benchmarks tracker — https://llm-tracker.info/RTX-PRO-6000

---

## Quick reference card (for an agent in a hurry)

```
PROD POD (NEVER TOUCH): u3qiwwpdxqaypp
GPU: 1× RTX Pro 6000 Blackwell, 96 GB, $1.70/hr
VOLUME: 1 TB at /workspace
PRECISION: bf16 (FP8 only for Llama 70B if used)
TP: 1 (single GPU)
LIFECYCLE: destroy after each phase, never stop
SDK: runpod (Python), NOT runpodctl shell-out
WRAPPER: src/runpod/lifecycle.py — only place destructive ops live
PATH ROOT: $DATA_ROOT (./data local, /workspace cloud)
TAG: project=multi-axis-persona-safety
PHASED: subject pod → destroy → judge pod → destroy → cross-check → destroy
ESTIMATED COST: $250-370 compute + $70-140 volume (vs $150 prior ceiling — RESURFACE TO USER)
```
