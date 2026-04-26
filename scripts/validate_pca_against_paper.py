"""Validate our centered PCA + projection helpers against the paper's released
bf16 vectors (CPU-only, no GPU needed).

Loads `data/paper_artifacts/assistant_axis_vectors/gemma-2-27b/`:
  - assistant_axis.pt  → (46, 4608) released AA
  - role_vectors/*.pt  → 275 files, each (46, 4608)

For each layer 0..45:
  1. Stack the 275 role vectors at that layer → (275, 4608)
  2. Fit centered PCA → PC1 direction (4608,)
  3. Compute cos_sim(PC1, AA[layer])
Pick L* = argmax. Assert cos_sim(PC1, AA) at L* > 0.71 per paper line 96.

Output:
  - prints per-layer cos_sim curve
  - prints L* and the cos_sim at L*
  - hard-fails if L* cos_sim < 0.71 (Stage 2 T2.2 acceptance criterion)

Usage:
  uv run python -m scripts.validate_pca_against_paper
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make src/ importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.pca import fit_pca  # noqa: E402
from src.analysis.projections import argmax_cos_sim_layer, cos_sim  # noqa: E402

PAPER_DIR = Path("data/paper_artifacts/assistant_axis_vectors/gemma-2-27b")
EXPECTED_THRESHOLD_AT_LSTAR = 0.71  # paper line 96
ALL_LAYERS_FLOOR = 0.60  # paper line 3426: > 0.60 at all layers


def main() -> None:
    aa = torch.load(PAPER_DIR / "assistant_axis.pt", weights_only=True)  # (46, 4608)
    n_layers, d_model = aa.shape
    print(f"AA shape: {tuple(aa.shape)} dtype={aa.dtype}")

    role_files = sorted((PAPER_DIR / "role_vectors").iterdir())
    print(f"Loading {len(role_files)} role vectors...")
    role_stack = torch.stack(
        [torch.load(p, weights_only=True) for p in role_files], dim=0
    )  # (275, 46, 4608)
    print(f"Role stack shape: {tuple(role_stack.shape)}")

    pc1_per_layer = np.zeros((n_layers, d_model), dtype=np.float32)
    sims = np.zeros(n_layers, dtype=np.float32)

    print("\nFitting per-layer PCA + computing cos_sim(PC1, AA)...")
    for layer in range(n_layers):
        role_at_layer = role_stack[:, layer, :].float().numpy()  # (275, 4608)
        result = fit_pca(role_at_layer)
        pc1 = result.components[0]  # (4608,) leading PC
        # PCA sign is arbitrary; align to AA
        s = cos_sim(pc1, aa[layer])
        if s < 0:
            pc1 = -pc1
            s = -s
        pc1_per_layer[layer] = pc1
        sims[layer] = s

    print(f"\nPer-layer cos_sim(PC1, AA):")
    for layer in range(n_layers):
        marker = "  *" if sims[layer] == sims.max() else "   "
        print(f"  layer {layer:2d}: {sims[layer]:.4f}{marker}")

    l_star = int(np.argmax(sims))
    s_at_lstar = float(sims[l_star])
    print(f"\nL* = {l_star}, cos_sim(PC1, AA) at L* = {s_at_lstar:.4f}")
    print(f"min cos_sim across all layers: {float(sims.min()):.4f}")
    print(f"max cos_sim across all layers: {float(sims.max()):.4f}")

    # Acceptance per Stage 2 T2.2 + paper conventions
    assert s_at_lstar > EXPECTED_THRESHOLD_AT_LSTAR, (
        f"cos_sim(PC1, AA) at L*={l_star} is {s_at_lstar:.4f} < paper threshold {EXPECTED_THRESHOLD_AT_LSTAR}. "
        "Extraction or PCA pipeline is broken."
    )
    if float(sims.min()) <= ALL_LAYERS_FLOOR:
        print(
            f"WARN: min cos_sim {float(sims.min()):.4f} <= paper's reported {ALL_LAYERS_FLOOR} "
            f"(line 3426). Likely fine if the dip is at very early or very late layers."
        )

    # Sanity: argmax_cos_sim_layer agrees
    pc1_tensor = torch.from_numpy(pc1_per_layer)
    l_check, s_check = argmax_cos_sim_layer(pc1_tensor, aa)
    assert l_check == l_star, f"argmax mismatch: {l_check} vs {l_star}"
    assert abs(s_check - s_at_lstar) < 1e-4
    print(f"\nargmax_cos_sim_layer agrees: L*={l_check}, cos={s_check:.4f}")

    print("\nT2.2 acceptance: PASS")
    print(f"Plan B will use L* = {l_star} for Gemma 2 27B extraction layer.")


if __name__ == "__main__":
    main()
