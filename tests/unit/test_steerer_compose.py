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


def test_sign_flip_makes_cap_role_positive() -> None:
    """The from_config sign-flip (steerer.py:73-105) negates Assistant-positive
    AA into role-positive direction expected by upstream `_apply_cap` ceiling.

    Rig: build a tiny linear-passthrough model (each block is identity) so the
    pre-hook activation equals the input. Inject input h such that <h, v_aa> = +1.0
    (Assistant-territory in our convention). After the sign-flip, upstream sees
    v_role = -v_aa, so <h, v_role> = -1.0.

    Caller passes cap_thresholds in role-positive space:
      - tau = -2.0 → upstream excess = max(-1.0 - (-2.0), 0) = 1.0 → fires.
      - tau = 0.0  → upstream excess = max(-1.0 - 0.0, 0)  = 0   → no fire.
    """
    d, n = 16, 4
    model = _TinyModel(d=d, n_layers=n)
    # Identity blocks so output equals input.
    for blk in model.model.layers:
        with torch.no_grad():
            blk.weight.copy_(torch.eye(d))

    # Pick v_aa as a unit vector; build h = +1.0 along v_aa.
    [v_aa] = _orthonormal(d, 1, seed=7)
    h = v_aa.unsqueeze(0).unsqueeze(0).clone()  # (1, 1, d), <h, v_aa> = 1.0

    # Case 1: tau = -2.0 (in role-positive space) → cap fires, h is modified.
    cfg_fires = SteeringConfig(
        vectors=[v_aa],
        intervention_type="capping",
        layer_indices=[(0, 0)],
        cap_thresholds=[-2.0],
    )
    with from_config(model, cfg_fires):
        out = model(h)
    # When fires: h <- h - v_role * 1.0 = h - (-v_aa) * 1.0 = h + v_aa.
    # So <out, v_aa> should be ~2.0 (was 1.0, gained 1.0).
    proj_after = (out.squeeze() @ v_aa).item()
    assert proj_after > 1.5, (
        f"cap-fires case: expected projection >1.5, got {proj_after:.3f}. "
        f"Sign flip in from_config may not be working."
    )

    # Case 2: tau = 0.0 (in role-positive space) → cap does NOT fire.
    cfg_no_fire = SteeringConfig(
        vectors=[v_aa],
        intervention_type="capping",
        layer_indices=[(0, 0)],
        cap_thresholds=[0.0],
    )
    with from_config(model, cfg_no_fire):
        out2 = model(h)
    proj_unchanged = (out2.squeeze() @ v_aa).item()
    assert abs(proj_unchanged - 1.0) < 0.05, (
        f"no-fire case: expected projection ~1.0, got {proj_unchanged:.3f}. "
        f"Cap fired when it shouldn't have."
    )


def test_plan_b_tau_lies_in_assistant_role_gap() -> None:
    """After Step 3 re-runs for capping layers [33,38], verify the role-positive
    tau (-p75 of role-rollout +AA projections) is below role-mean and above a
    sensible Assistant lower bound. CI-safe: skips if file missing."""
    import json
    from pathlib import Path

    tau_path = Path("results/plan_b_gemma2_27b/extraction/tau_calibration.json")
    if not tau_path.exists():
        import pytest

        pytest.skip(f"{tau_path} missing — Step 3 not yet re-run with new layer range")

    payload = json.loads(tau_path.read_text())
    expected_layers = {33, 34, 35, 36, 37, 38}
    actual_layers = {int(k) for k in payload["per_layer"].keys()}
    if not expected_layers.issubset(actual_layers):
        import pytest

        pytest.skip(
            f"tau_calibration.json missing layers {expected_layers - actual_layers}; "
            "re-run Step 3 with new capping range first."
        )

    for L in expected_layers:
        per = payload["per_layer"][str(L)]
        tau_role = -float(per["p75"])  # role-positive tau
        mean_role_pos = -float(per["mean"])  # role-mean in role-positive space
        # Tau should be MORE negative than role-mean (i.e., closer to Assistant).
        assert tau_role < mean_role_pos, (
            f"L={L}: tau_role={tau_role:.2f} should be < mean_role_pos={mean_role_pos:.2f}. "
            "tau is in role territory, not in the Assistant-role gap."
        )
        # Slack lower bound: tau shouldn't dive below -10000 (projections grow with depth
        # but should stay within typical Gemma 2 27B residual norms).
        assert tau_role > -10000, (
            f"L={L}: tau_role={tau_role:.2f} dove below -10000; sanity-check tau magnitude."
        )
