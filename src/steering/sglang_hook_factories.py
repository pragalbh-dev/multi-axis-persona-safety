"""Hook factories for SGLang's `--forward-hooks` mechanism.

Mirrors the math in `external/assistant-axis::ActivationSteering._apply_*`
exactly, so SGLang-served subjects produce equivalent activations to the HF
reference path. SGLang loads each spec via `register_forward_hooks(...)` in
`hook_manager.py`; the spec format is:

    {"target_modules": [glob],
     "hook_factory": "src.steering.sglang_hook_factories:<name>",
     "name": str,
     "config": dict}

Each factory below has signature `factory(config: dict) -> hook_callable`,
where `hook_callable(module, inputs, output)` returns the modified output (or
the unchanged output) following PyTorch's `register_forward_hook` contract.

Vectors are loaded from `safetensors` files referenced in `config["vector_path"]`.
For ordering with `cap_and_steer`: register cap factory first, steer factory
second in the `--forward-hooks` JSON list (PyTorch fires in registration order).

See `plans/sglang_post_plan_b_spike.md` Sections "Hook ordering" and
"Tuple-vs-tensor output" for the design.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


_HOOK_CALL_COUNTERS: dict[str, int] = {}
# No threading.Lock — `with lock: ...` breaks torch.compile (piecewise CUDA
# graph), and the GIL keeps `dict[k] = dict.get(k, 0) + 1` atomic enough for
# a debug counter. Race conditions here only mean a slightly off count, which
# is fine for the decode-coverage assertion (looking for orders of magnitude).


def reset_call_counters() -> None:
    _HOOK_CALL_COUNTERS.clear()


def get_call_counters() -> dict[str, int]:
    return dict(_HOOK_CALL_COUNTERS)


def _bump(name: str) -> None:
    _HOOK_CALL_COUNTERS[name] = _HOOK_CALL_COUNTERS.get(name, 0) + 1


def _load_vector(path: str | Path, key: str = "vector") -> torch.Tensor:
    """Load a 1-D direction vector from a safetensors file under `key`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"vector file not found: {p}")
    tensors = load_file(str(p))
    if key not in tensors:
        raise KeyError(f"key {key!r} not in {p}; available: {list(tensors)}")
    v = tensors[key]
    if v.ndim != 1:
        raise ValueError(f"vector at {p}/{key} must be 1-D, got shape {tuple(v.shape)}")
    return v


def _split_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    """Match `src/extraction/backend_hf.py:328` tuple-vs-tensor handling."""
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, (tuple, list)):
        if not torch.is_tensor(output[0]):
            raise TypeError(f"hook output[0] is not a tensor: {type(output[0])}")
        rest = tuple(output[1:])
        return output[0], rest
    raise TypeError(f"unsupported hook output type: {type(output)}")


def _rejoin_output(t: torch.Tensor, rest: tuple[Any, ...] | None) -> Any:
    if rest is None:
        return t
    return (t, *rest)


def _apply_addition(
    h: torch.Tensor,
    v: torch.Tensor,
    coeff: float,
    positions: str,
) -> torch.Tensor:
    """Mirror `ActivationSteering._apply_addition` (steering.py:278-288).

    Handles both 3D `(B, L, d)` (HF/eager forward) and 2D `(N, d)`
    (SGLang continuous batching where the (B, L) axes are concatenated).
    For positions="last" with 2D input we apply to all rows — SGLang doesn't
    expose per-sequence boundaries inside the hook, so callers should use
    positions="all" with SGLang.
    """
    v = v.to(device=h.device, dtype=h.dtype)
    steer = coeff * v
    if h.ndim == 2:
        # (N, d): apply to every token; positions="last" not meaningful here.
        return h + steer
    if positions == "all":
        return h + steer
    out = h.clone()
    out[:, -1, :] += steer
    return out


def _apply_cap(
    h: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    positions: str,
) -> torch.Tensor:
    """Mirror `ActivationSteering._apply_cap` (steering.py:317-332).

    Formula: h ← h - v_unit · max(<h, v_unit> - tau, 0)
    where v_unit = v / ‖v‖.

    Handles both 3D `(B, L, d)` (HF/eager forward) and 2D `(N, d)`
    (SGLang continuous batching). See `_apply_addition` note about
    positions="last" with 2D input.
    """
    v = v.to(device=h.device, dtype=h.dtype)
    v_unit = v / (v.norm() + 1e-8)
    if h.ndim == 2:
        # (N, d): treat each token independently.
        proj = h @ v_unit  # (N,)
        excess = (proj - tau).clamp(min=0.0)
        return h - excess.unsqueeze(-1) * v_unit
    if positions == "all":
        proj = torch.einsum("bld,d->bl", h, v_unit)
        excess = (proj - tau).clamp(min=0.0)
        return h - torch.einsum("bl,d->bld", excess, v_unit)
    out = h.clone()
    last = out[:, -1, :]
    proj = torch.einsum("bd,d->b", last, v_unit)
    excess = (proj - tau).clamp(min=0.0)
    out[:, -1, :] = last - torch.einsum("b,d->bd", excess, v_unit)
    return out


def addition_factory(config: dict[str, Any]) -> Callable[..., Any]:
    """Build an addition hook: h ← h + λv at the configured positions.

    config keys:
      vector_path (str)   — safetensors path
      vector_key  (str)   — optional, default "vector"
      coefficient (float) — λ
      positions   (str)   — "all" or "last", default "all"
      name        (str)   — optional, used for the call counter
    """
    v = _load_vector(config["vector_path"], config.get("vector_key", "vector"))
    coeff = float(config["coefficient"])
    positions = config.get("positions", "all")
    name = config.get("name", "addition")

    def hook(module, inputs, output):  # noqa: ANN001
        _bump(name)
        h, rest = _split_output(output)
        h_new = _apply_addition(h, v, coeff, positions)
        return _rejoin_output(h_new, rest)

    return hook


def capping_factory(config: dict[str, Any]) -> Callable[..., Any]:
    """Build a capping hook: h ← h - v·max(<h,v_unit>-τ, 0).

    Replicates the negation fix in `src/steering/steerer.py:86-94`: if
    `negate_vector=True` (default for compatibility with our pipeline that
    stores Assistant-positive AA), the loaded vector is flipped before use.

    config keys:
      vector_path     (str)   — safetensors path
      vector_key      (str)   — optional, default "vector"
      tau             (float) — τ threshold (in the v-positive direction used by the cap)
      positions       (str)   — "all" or "last", default "all"
      negate_vector   (bool)  — default True; matches `from_config` behavior
      name            (str)   — optional, used for the call counter
    """
    v = _load_vector(config["vector_path"], config.get("vector_key", "vector"))
    if config.get("negate_vector", True):
        v = -v
    tau = float(config["tau"])
    positions = config.get("positions", "all")
    name = config.get("name", "capping")

    def hook(module, inputs, output):  # noqa: ANN001
        _bump(name)
        h, rest = _split_output(output)
        h_new = _apply_cap(h, v, tau, positions)
        return _rejoin_output(h_new, rest)

    return hook


def cap_and_steer_factory(config: dict[str, Any]) -> Callable[..., Any]:
    """Composed cap+steer in one hook (cap then steer, matching HF order).

    Preferred path is to register two separate hooks via the --forward-hooks
    JSON list (cap entry first, steer entry second). This factory exists so
    the spike can compare a single-fused hook against the two-hook composition.

    config keys:
      cap   (dict) — same shape as capping_factory config
      steer (dict) — same shape as addition_factory config
      name  (str)  — optional, used for the call counter
    """
    cap_cfg = config["cap"]
    steer_cfg = config["steer"]

    v_cap = _load_vector(cap_cfg["vector_path"], cap_cfg.get("vector_key", "vector"))
    if cap_cfg.get("negate_vector", True):
        v_cap = -v_cap
    tau = float(cap_cfg["tau"])
    cap_pos = cap_cfg.get("positions", "all")

    v_steer = _load_vector(steer_cfg["vector_path"], steer_cfg.get("vector_key", "vector"))
    coeff = float(steer_cfg["coefficient"])
    steer_pos = steer_cfg.get("positions", "all")
    name = config.get("name", "cap_and_steer")

    def hook(module, inputs, output):  # noqa: ANN001
        _bump(name)
        h, rest = _split_output(output)
        h = _apply_cap(h, v_cap, tau, cap_pos)
        h = _apply_addition(h, v_steer, coeff, steer_pos)
        return _rejoin_output(h, rest)

    return hook


def multi_axis_cap_factory(config: dict[str, Any]) -> Callable[..., Any]:
    """N caps on the same module, applied sequentially.

    config keys:
      axes (list[dict]) — each entry is a capping_factory config (vector_path, tau, ...)
      positions (str)   — applied to all axes if not set per-axis, default "all"
      name (str)        — optional, used for the call counter
    """
    default_pos = config.get("positions", "all")
    axes = []
    for entry in config["axes"]:
        v = _load_vector(entry["vector_path"], entry.get("vector_key", "vector"))
        if entry.get("negate_vector", True):
            v = -v
        tau = float(entry["tau"])
        pos = entry.get("positions", default_pos)
        axes.append((v, tau, pos))
    name = config.get("name", "multi_axis_cap")

    def hook(module, inputs, output):  # noqa: ANN001
        _bump(name)
        h, rest = _split_output(output)
        for v, tau, pos in axes:
            h = _apply_cap(h, v, tau, pos)
        return _rejoin_output(h, rest)

    return hook


__all__ = [
    "addition_factory",
    "capping_factory",
    "cap_and_steer_factory",
    "multi_axis_cap_factory",
    "reset_call_counters",
    "get_call_counters",
]
