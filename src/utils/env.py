"""Project-wide environment setup.

Import this module at the top of every script or notebook to:
  1. Pin CUDA_VISIBLE_DEVICES=0,1,2,3 before torch / vLLM touches CUDA.
  2. Load .env (HF_TOKEN etc.) via python-dotenv.
  3. Expose a seed-everything helper.

All 4 GPUs on the box are available to this project (revert from Stage 0's
2-GPU constraint; see plans/decisions.md 2026-04-25 fp8->bf16 entry).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# Pin GPUs BEFORE importing torch anywhere in the process.
# Respect an existing narrower override (e.g. CUDA_VISIBLE_DEVICES=0,1 for a
# specific test) but reject anything that mentions a non-existent GPU index.
_ALLOWED = {"0", "1", "2", "3"}
_existing = os.environ.get("CUDA_VISIBLE_DEVICES")
if _existing is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
else:
    requested = {x.strip() for x in _existing.split(",") if x.strip()}
    if not requested.issubset(_ALLOWED):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={_existing!r} requests GPUs outside the allowed set {_ALLOWED}."
        )

# Load .env (if present) before anything reads HF_TOKEN.
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch, and torch.cuda. Also sets PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
