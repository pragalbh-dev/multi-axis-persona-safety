"""Stage 1 T1.3 — steerer wrapper composition.

Uses a tiny synthetic nn.Module so we don't load a real LLM. Verifies
hook-registration counts for capping range expansion + cap_and_steer
composition + multi_axis_cap orthogonality check.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.steering import (
    SteeringConfig,
    cap_and_steer,
    from_config,
    multi_axis_cap,
    verify_orthogonality,
)


class _Inner(nn.Module):
    def __init__(self, d: int, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n)])


class _TinyModel(nn.Module):
    def __init__(self, d: int, n_layers: int) -> None:
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": d})  # type: ignore[assignment]
        self.model = _Inner(d, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.model.layers:
            x = blk(x)
        return x


def _orthonormal(d: int, n: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, n, generator=g)
    Q, _ = torch.linalg.qr(M)
    return [Q[:, i].clone() for i in range(n)]


def test_capping_range_expansion_registers_one_hook_per_layer() -> None:
    model = _TinyModel(d=32, n_layers=8)
    [v] = _orthonormal(32, 1)
    cfg = SteeringConfig(
        vectors=[v],
        intervention_type="capping",
        layer_indices=[(2, 5)],
        cap_thresholds=[0.5],
    )
    with from_config(model, cfg) as steerer:
        # Range [2, 5] inclusive expands to layers 2,3,4,5 → 4 unique hooks.
        assert len(steerer._handles) == 4


def test_cap_and_steer_registers_both() -> None:
    model = _TinyModel(d=32, n_layers=8)
    [v1, v2] = _orthonormal(32, 2)
    cap_cfg = SteeringConfig(
        vectors=[v1],
        intervention_type="capping",
        layer_indices=[(2, 5)],
        cap_thresholds=[0.5],
    )
    steer_cfg = SteeringConfig(
        vectors=[v2],
        intervention_type="addition",
        layer_indices=[3],
        coefficients=[1.0],
    )
    with cap_and_steer(model, cap_cfg, steer_cfg) as (cap, steer):
        assert len(cap._handles) == 4
        assert len(steer._handles) == 1


def test_multi_axis_cap_orthogonal_ok() -> None:
    model = _TinyModel(d=32, n_layers=8)
    v1, v2, v3 = _orthonormal(32, 3)
    ms = multi_axis_cap(model, [(v1, (2, 5), 0.5), (v2, (2, 5), 0.4), (v3, (2, 5), 0.3)])
    with ms as steerer:
        # 4 unique layers × 3 vectors but `vectors_by_layer` groups by layer.
        assert len(steerer._handles) == 4


def test_multi_axis_cap_warns_on_non_orthogonal() -> None:
    model = _TinyModel(d=32, n_layers=8)
    [v1, v2] = _orthonormal(32, 2)
    v_bad = (v1 + v2).clone()
    v_bad = v_bad / v_bad.norm()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        multi_axis_cap(model, [(v1, (2, 5), 0.5), (v_bad, (2, 5), 0.4)])
        assert any("orthogonality" in str(w.message) for w in wlist)


def test_verify_orthogonality_detects_collinear() -> None:
    [v] = _orthonormal(16, 1)
    ok, sims = verify_orthogonality([v, v.clone()])
    assert not ok
    assert sims[0, 1].item() > 0.9
