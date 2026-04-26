"""Per-axis tau calibration for multi-axis defence (Phase 4).

For each candidate cap direction (sign-matched PC2/PC3, v_harm), compute the
threshold tau such that capping the projection along that direction at tau
matches the paper's sign convention (clamp role-positive overshoot back toward
the safe end).

Background: Plan B step 3 calibrates AA's tau via -p75 of <h, +AA> projections
on role rollouts (role-positive space). The sign-flip math in
external/assistant-axis::ActivationSteering._apply_cap subtracts excess in the
direction the user passed (negated AA in our pipeline). For PC2/PC3/v_harm
analogously, we need:
  tau_axis = p_PCT of <h, sign_axis · axis_unit> on role rollouts at each capping layer
where sign_axis is chosen to make role-positive = the direction we want to
clamp from above. By convention we use the sign that matches v_harm's
direction (so positive projection = harm-toward direction).

For PC_i: sign_i = sign(cos(v_harm, PC_i))  →  axis_unit = sign_i · PC_i_unit
For v_harm: sign = +1, axis_unit = v_harm_unit
For AA: sign = -1 if cos(v_harm, AA) < 0 (Plan B convention), or use Plan B's
  precomputed taus directly.

This module is CPU-only. Loads cached role activations from data/cache/ and
extraction artifacts, writes tau JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from src.extraction.types import ActivationCache


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n


def calibrate_axis_tau(
    *,
    role_acts_per_layer: dict[int, np.ndarray],  # layer → (n, d) activations
    axis_unit: np.ndarray,                        # (d,) the role-positive direction we'll cap above
    percentile: float = 25.0,
) -> dict[str, dict[str, float]]:
    """Per-layer tau distribution (p1, p10, p25, p50, p75, mean, n_samples)."""
    out: dict[str, dict[str, float]] = {}
    for layer, acts in role_acts_per_layer.items():
        proj = acts @ axis_unit
        out[str(layer)] = {
            "p1":  float(np.quantile(proj, 0.01)),
            "p10": float(np.quantile(proj, 0.10)),
            "p25": float(np.quantile(proj, 0.25)),
            "p50": float(np.quantile(proj, 0.50)),
            "p75": float(np.quantile(proj, 0.75)),
            "mean": float(proj.mean()),
            "n_samples": int(proj.size),
        }
    return out


def main() -> None:
    """CLI: calibrate taus for sign-matched PC2/PC3/PC4 + v_harm at the capping layers."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--harm-direction-json",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/harm_direction_merged.json"),
    )
    ap.add_argument(
        "--extraction-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extraction"),
    )
    ap.add_argument(
        "--role-cache-root",
        type=Path,
        default=Path("data/cache/activations/gemma_2_27b/plan_b_role_rollouts"),
        help="dir with L<layer>.safetensors files for role rollouts",
    )
    ap.add_argument(
        "--capping-layers",
        type=int,
        nargs="+",
        default=[33, 34, 35, 36, 37, 38],
    )
    ap.add_argument("--pc-indices", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--include-v-harm", action="store_true", default=True)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/tau_per_axis.json"),
    )
    args = ap.parse_args()

    # Load harm direction info to get sign-matched axes.
    hd = json.loads(args.harm_direction_json.read_text())
    cos_pcs = hd["cos_sim_v_harm_pcs"]

    # Load PCs at L*=21 (we recompute at each capping layer; the role-vector PCA
    # was on L*=21 means but we project capping-layer activations onto the
    # *L*-derived* PC direction. This matches Plan B's convention for AA: AA
    # is computed at L*=21 then projected onto activations at every capping
    # layer.) Actually, no — we need to project onto a layer-specific direction.
    # The paper's AA is layer-specific; aa.safetensors has per-layer AA.
    # PCs are L*-specific only. For PC capping at deeper layers we'd need
    # PC-per-layer, but that's not what Plan B did. Plan B used L*-PC for
    # projection at capping layers (treating role-PCA as a global subspace).
    # We follow the same convention.
    pcs_at_lstar = load_file(str(args.extraction_dir / "pcs.safetensors"))["pcs_at_lstar"].float().numpy()
    pcs_unit = np.stack([_unit(pcs_at_lstar[i]) for i in range(pcs_at_lstar.shape[0])])

    # Compute v_harm direction. Need to recompute since we don't store it.
    # Use Plan B baseline + Phase 1 extended baseline activations at L*=21,
    # filtered to baseline rows, split by harm_binary.
    import pandas as pd
    plan_b_details = Path("results/plan_b_gemma2_27b/details.parquet")
    ext_details = Path("results/plan_b_gemma2_27b/extensions/baseline_extended.parquet")
    plan_b_cache = ActivationCache.load(
        Path("data/cache/activations/gemma_2_27b/plan_b_per_prompt_L21/L21")
    )
    ext_cache = ActivationCache.load(
        Path("data/cache/activations/gemma_2_27b/baseline_extended_L21/L21")
    )
    plan_b_lookup = {p: i for i, p in enumerate(plan_b_cache.prompt_ids)}
    ext_lookup = {p: i for i, p in enumerate(ext_cache.prompt_ids)}

    plan_b_df = pd.read_parquet(plan_b_details)
    plan_b_baseline = plan_b_df[plan_b_df["condition_id"] == "baseline"]
    ext_df = pd.read_parquet(ext_details)
    plan_b_acts = plan_b_cache.tensor.float().numpy()
    ext_acts = ext_cache.tensor.float().numpy()

    acts_list, harm_list = [], []
    for _, row in plan_b_baseline.iterrows():
        key = f"{row['prompt_id']}::baseline"
        if key in plan_b_lookup:
            acts_list.append(plan_b_acts[plan_b_lookup[key]])
            harm_list.append(int(row["harm_binary"]))
    for _, row in ext_df.iterrows():
        key = f"{row['prompt_id']}::baseline"
        if key in ext_lookup:
            acts_list.append(ext_acts[ext_lookup[key]])
            harm_list.append(int(row["harm_binary"]))
    acts_arr = np.stack(acts_list)
    harm_arr = np.array(harm_list, dtype=np.int32)
    v_harm = acts_arr[harm_arr == 1].mean(0) - acts_arr[harm_arr == 0].mean(0)
    v_harm_unit_lstar = _unit(v_harm)

    print(f"Computed v_harm at L*=21 from n={len(harm_arr)} baseline rows ({int(harm_arr.sum())} harm)")

    # Load role activations at each capping layer.
    role_acts_per_layer = {}
    for layer in args.capping_layers:
        cache = ActivationCache.load(args.role_cache_root / f"L{layer}")
        role_acts_per_layer[layer] = cache.tensor.float().numpy()
        print(f"  L{layer}: {role_acts_per_layer[layer].shape}")

    # For each axis: build sign-matched unit direction at L*=21, then project
    # the deeper-layer activations onto that L*-direction. (This is the same
    # subspace assumption as Plan B used for AA.)
    output: dict[str, dict] = {
        "capping_layers": args.capping_layers,
        "axes": {},
    }
    for pc_idx in args.pc_indices:
        sign_i = -1 if cos_pcs[pc_idx - 1] < 0 else 1
        axis_unit = sign_i * pcs_unit[pc_idx - 1]
        print(f"signmatched_pc{pc_idx}: cos(v_harm,PC{pc_idx})={cos_pcs[pc_idx-1]:+.3f} → sign={sign_i:+d}")
        per_layer = calibrate_axis_tau(
            role_acts_per_layer=role_acts_per_layer,
            axis_unit=axis_unit,
        )
        output["axes"][f"signmatched_pc{pc_idx}"] = {
            "sign": sign_i,
            "cos_v_harm_pc": float(cos_pcs[pc_idx - 1]),
            "per_layer": per_layer,
        }

    if args.include_v_harm:
        per_layer = calibrate_axis_tau(
            role_acts_per_layer=role_acts_per_layer,
            axis_unit=v_harm_unit_lstar,
        )
        output["axes"]["v_harm"] = {
            "sign": 1,
            "per_layer": per_layer,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n→ {args.output}")
    # Show p25 (the cap threshold we'll use) per axis per layer
    print("\np25 per axis per layer (the cap threshold candidate):")
    for ax, info in output["axes"].items():
        layers = sorted(int(k) for k in info["per_layer"].keys())
        p25s = [info["per_layer"][str(l)]["p25"] for l in layers]
        print(f"  {ax}: {dict(zip(layers, [round(x, 2) for x in p25s]))}")


if __name__ == "__main__":
    main()
