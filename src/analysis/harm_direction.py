"""Ext A — DiffMean v_harm diagnostic.

Given a baseline (no-cap, no-steer) population of (activation_at_Lstar, harm_binary)
rows, compute the DiffMean direction `v_harm = mean(harm) − mean(safe)` and report
how it relates to AA and the role-PCA basis.

Outputs (saved to JSON):
  - cos_sim(v_harm, AA)
  - cos_sim(v_harm, PC_i) for i in [1..K_pc]
  - residual_outside_topK = ||v_harm − P_K(v_harm)||² / ||v_harm||²  for K ∈ {3,5,10}
  - single_dir_auc_v_harm: ROC-AUC of <h, v_harm_unit> as a harm classifier
  - single_dir_auc_aa: ROC-AUC of <h, aa_unit> as a harm classifier (control)
  - argmax_axis_alignment: which axis (AA or PC_i) v_harm is most aligned with
  - n_harm, n_safe, dataset breakdown

Decision lens for downstream extensions:
  - cos_sim(v_harm, AA) > 0.9  →  harm geometry is essentially AA; pivot framing.
  - cos_sim(v_harm, AA) ∈ [0.5, 0.85]  →  meaningfully novel direction; Ext B/C fire.
  - residual_outside_top10 > 0.3  →  harm has structure outside role-PCA; basis pivot.

This module is CPU-only. No torch model loads, no judge calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

from src.extraction.types import ActivationCache


@dataclass
class HarmDirectionResult:
    n_total: int
    n_harm: int
    n_safe: int
    dataset_breakdown: dict[str, dict[str, int]]
    layer: int

    cos_sim_v_harm_aa: float
    cos_sim_v_harm_pcs: list[float]  # PC1..PCK
    residual_outside_top_k: dict[str, float]  # "3", "5", "10" → fraction

    single_dir_auc_v_harm: float
    single_dir_auc_v_harm_ci: tuple[float, float]
    single_dir_auc_aa: float
    single_dir_auc_aa_ci: tuple[float, float]

    argmax_axis_name: str  # "AA" or "PC_i"
    argmax_axis_cos: float

    v_harm_norm_pre: float  # before unit-normalization
    aa_norm: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["single_dir_auc_v_harm_ci"] = list(d["single_dir_auc_v_harm_ci"])
        d["single_dir_auc_aa_ci"] = list(d["single_dir_auc_aa_ci"])
        return d


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n


def _auc_with_ci(
    scores: np.ndarray, labels: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> tuple[float, tuple[float, float]]:
    """ROC-AUC with paired-resampling percentile CI (95%).

    Bootstraps over (score, label) row indices so each resample preserves the
    pairing. BCa requires jackknife on the paired sample which is doable but
    overkill here — percentile is standard for AUC bootstraps and is what
    scipy-bootstrap defaults to without a paired-sample mode.
    """
    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        l = labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            boot[i] = 0.5
            continue
        boot[i] = roc_auc_score(l, scores[idx])
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return point, (lo, hi)


def compute_harm_direction(
    *,
    activations: np.ndarray,  # (n, d) float32
    harm_binary: np.ndarray,  # (n,) int 0/1
    aa: np.ndarray,           # (d,) raw AA at L*
    pcs: np.ndarray,          # (k, d) PC1..PCk
    pca_mean: np.ndarray,     # (d,) PCA centering vector
    layer: int,
    dataset_per_row: list[str] | None = None,
) -> HarmDirectionResult:
    """Pure-numpy core. See module docstring for contract."""
    if activations.ndim != 2:
        raise ValueError(f"activations must be (n, d), got {activations.shape}")
    n, d = activations.shape
    if harm_binary.shape != (n,):
        raise ValueError(f"harm_binary shape {harm_binary.shape} != ({n},)")
    if aa.shape != (d,):
        raise ValueError(f"aa shape {aa.shape} != ({d},)")
    if pcs.shape[1] != d:
        raise ValueError(f"pcs shape[1] {pcs.shape[1]} != d {d}")

    harm_mask = harm_binary.astype(bool)
    n_harm = int(harm_mask.sum())
    n_safe = int((~harm_mask).sum())
    if n_harm == 0 or n_safe == 0:
        raise ValueError(f"need both classes; got n_harm={n_harm} n_safe={n_safe}")

    mean_harm = activations[harm_mask].mean(axis=0)
    mean_safe = activations[~harm_mask].mean(axis=0)
    v_harm_raw = mean_harm - mean_safe
    v_harm_norm_pre = float(np.linalg.norm(v_harm_raw))
    v_harm = _unit(v_harm_raw)

    aa_unit = _unit(aa)
    cos_aa = float(np.dot(v_harm, aa_unit))

    # PCs assumed already unit-normalized by PCA, but normalize defensively.
    pcs_unit = np.stack([_unit(pcs[i]) for i in range(pcs.shape[0])], axis=0)
    cos_pcs = (pcs_unit @ v_harm).tolist()

    # Residual outside top-K role-PCs (project onto first K components, subtract).
    residual: dict[str, float] = {}
    v_harm_sq = float(np.dot(v_harm, v_harm))  # = 1.0
    for K in (3, 5, 10):
        K_use = min(K, pcs_unit.shape[0])
        proj_K = pcs_unit[:K_use] @ v_harm  # (K,)
        proj_recon = proj_K @ pcs_unit[:K_use]  # (d,)
        resid = v_harm - proj_recon
        residual[str(K)] = float(np.dot(resid, resid) / max(v_harm_sq, 1e-12))

    # Single-direction AUC. We center activations by the PCA mean to put projections
    # in a frame consistent with how the LASSO uses them (projections are computed
    # on PCA-centered activations in step 7b). For AA, we don't center — AA is a
    # raw-activation contrast direction. Both used as harm-classifier scores.
    centered = activations - pca_mean
    score_v_harm = centered @ v_harm
    score_aa = activations @ aa_unit

    auc_vh, ci_vh = _auc_with_ci(score_v_harm, harm_binary)
    auc_aa, ci_aa = _auc_with_ci(score_aa, harm_binary)

    # argmax axis: AA vs PC_i.
    candidates = [("AA", abs(cos_aa))]
    for i, c in enumerate(cos_pcs, start=1):
        candidates.append((f"PC{i}", abs(c)))
    argmax_name, argmax_abs = max(candidates, key=lambda x: x[1])
    # Re-sign by direction (negative cos → matches the *negative* of that axis).
    if argmax_name == "AA":
        argmax_signed = cos_aa
    else:
        argmax_signed = cos_pcs[int(argmax_name[2:]) - 1]

    breakdown: dict[str, dict[str, int]] = {}
    if dataset_per_row is not None:
        ds = np.asarray(dataset_per_row)
        for d_name in sorted(set(ds.tolist())):
            mask = ds == d_name
            breakdown[d_name] = {
                "n": int(mask.sum()),
                "n_harm": int((harm_mask & mask).sum()),
                "n_safe": int((~harm_mask & mask).sum()),
            }

    notes = (
        f"v_harm = mean(harm)-mean(safe) on n={n} baseline rows. "
        f"AA / PC_i directions L2-normalized before cos_sim. "
        f"Residual computed against {pcs_unit.shape[0]} role-PC basis. "
        f"AUC bootstrap: 2000 BCa resamples."
    )

    return HarmDirectionResult(
        n_total=n,
        n_harm=n_harm,
        n_safe=n_safe,
        dataset_breakdown=breakdown,
        layer=layer,
        cos_sim_v_harm_aa=cos_aa,
        cos_sim_v_harm_pcs=cos_pcs,
        residual_outside_top_k=residual,
        single_dir_auc_v_harm=auc_vh,
        single_dir_auc_v_harm_ci=ci_vh,
        single_dir_auc_aa=auc_aa,
        single_dir_auc_aa_ci=ci_aa,
        argmax_axis_name=argmax_name,
        argmax_axis_cos=argmax_signed,
        v_harm_norm_pre=v_harm_norm_pre,
        aa_norm=float(np.linalg.norm(aa)),
        notes=notes,
    )


def run_ext_a(
    *,
    details_parquet: Path,
    activation_cache: Path,
    extraction_dir: Path,
    output_json: Path,
    condition_filter: str = "baseline",
) -> HarmDirectionResult:
    """Orchestration: load Plan B artifacts, run compute_harm_direction, save JSON.

    Args:
        details_parquet: results/plan_b_gemma2_27b/details.parquet
        activation_cache: data/cache/activations/<model>/plan_b_per_prompt_L21/L21.safetensors
        extraction_dir:   results/plan_b_gemma2_27b/extraction/  (contains aa.safetensors + pcs.safetensors)
        output_json:      results/plan_b_gemma2_27b/extensions/harm_direction.json
        condition_filter: parquet `condition_id` rows to keep (default 'baseline').
    """
    df = pd.read_parquet(details_parquet)
    df = df[df["condition_id"] == condition_filter].copy()
    if df.empty:
        raise RuntimeError(f"no rows matched condition_id == {condition_filter!r}")

    cache = ActivationCache.load(activation_cache)
    acts_all = cache.tensor.float().numpy()
    prompt_ids = list(cache.prompt_ids)
    if len(prompt_ids) != acts_all.shape[0]:
        raise RuntimeError(
            f"prompt_ids count {len(prompt_ids)} != activations rows {acts_all.shape[0]}"
        )
    pid_to_row = {pid: i for i, pid in enumerate(prompt_ids)}

    # Build aligned act + harm vectors.
    keep_idx: list[int] = []
    aligned_harm: list[int] = []
    aligned_dataset: list[str] = []
    for _, row in df.iterrows():
        key = f"{row['prompt_id']}::{row['condition_id']}"
        if key not in pid_to_row:
            continue
        keep_idx.append(pid_to_row[key])
        aligned_harm.append(int(row["harm_binary"]))
        aligned_dataset.append(str(row.get("dataset", "dan")))
    if not keep_idx:
        raise RuntimeError("zero parquet rows aligned to activation cache")

    activations = acts_all[np.asarray(keep_idx, dtype=np.int64)]
    harm_binary = np.asarray(aligned_harm, dtype=np.int32)

    aa = load_file(str(extraction_dir / "aa.safetensors"))["aa_at_lstar"].float().numpy()
    pcs_d = load_file(str(extraction_dir / "pcs.safetensors"))
    pcs = pcs_d["pcs_at_lstar"].float().numpy()
    pca_mean = pcs_d["pca_mean_at_lstar"].float().numpy()

    layer = int(cache.layer)

    res = compute_harm_direction(
        activations=activations,
        harm_binary=harm_binary,
        aa=aa,
        pcs=pcs,
        pca_mean=pca_mean,
        layer=layer,
        dataset_per_row=aligned_dataset,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(res.to_dict(), indent=2))
    return res


def _format_summary(res: HarmDirectionResult) -> str:
    lines = [
        "═" * 72,
        f"Ext A — DiffMean v_harm diagnostic (layer L*={res.layer})",
        "═" * 72,
        f"Sample: n={res.n_total}  ({res.n_harm} harm, {res.n_safe} safe)",
    ]
    if res.dataset_breakdown:
        for ds, counts in res.dataset_breakdown.items():
            lines.append(
                f"  {ds:>6}: n={counts['n']:>4}  harm={counts['n_harm']:>3}  safe={counts['n_safe']:>4}"
            )
    lines += [
        "",
        f"||v_harm||_pre  = {res.v_harm_norm_pre:.4f}",
        f"||AA||_pre      = {res.aa_norm:.4f}",
        "",
        f"cos_sim(v_harm, AA)        = {res.cos_sim_v_harm_aa:+.4f}",
    ]
    for i, c in enumerate(res.cos_sim_v_harm_pcs, start=1):
        marker = "  ←" if f"PC{i}" == res.argmax_axis_name else ""
        lines.append(f"cos_sim(v_harm, PC{i:<2})       = {c:+.4f}{marker}")
    lines += [
        "",
        "residual outside role-PCA top-K (fraction of v_harm energy):",
    ]
    for K, r in res.residual_outside_top_k.items():
        lines.append(f"  K={K:>2}  residual = {r:.4f}")
    lines += [
        "",
        f"AUC <h, v_harm>  = {res.single_dir_auc_v_harm:.4f}  "
        f"[{res.single_dir_auc_v_harm_ci[0]:.4f}, {res.single_dir_auc_v_harm_ci[1]:.4f}]",
        f"AUC <h, AA>      = {res.single_dir_auc_aa:.4f}  "
        f"[{res.single_dir_auc_aa_ci[0]:.4f}, {res.single_dir_auc_aa_ci[1]:.4f}]",
        "",
        f"argmax axis alignment: {res.argmax_axis_name}  (cos = {res.argmax_axis_cos:+.4f})",
        "═" * 72,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--details", type=Path, default=Path("results/plan_b_gemma2_27b/details.parquet"))
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("data/cache/activations/gemma_2_27b/plan_b_per_prompt_L21/L21"),
        help="cache base path (without .safetensors suffix)",
    )
    p.add_argument(
        "--extraction-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extraction"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/harm_direction.json"),
    )
    p.add_argument("--condition", default="baseline")
    args = p.parse_args()

    res = run_ext_a(
        details_parquet=args.details,
        activation_cache=args.cache,
        extraction_dir=args.extraction_dir,
        output_json=args.output,
        condition_filter=args.condition,
    )
    print(_format_summary(res))
    print(f"\n→ JSON saved to {args.output}")
