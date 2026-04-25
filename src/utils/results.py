"""Result-directory contract enforcer.

Every experiment writes to `results/exp{N}_{name}/` and that directory MUST
contain a fixed set of files (see CONVENTIONS "Data & results layout"). This
module owns the create/resume logic so experiment scripts never roll their own.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.utils.config import ExperimentConfig, dump_experiment_config
from src.utils.manifest import (
    SCHEMA_VERSION,
    Manifest,
    current_git_sha,
    is_resumable,
    read_manifest,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def init_results_dir(cfg: ExperimentConfig) -> tuple[Path, Manifest]:
    """Create (or resume) the results directory for `cfg`.

    Returns `(results_dir, manifest)`. If the directory is resumable and
    `cfg.fresh` is False, the existing manifest is returned. Otherwise a fresh
    manifest is written and `config.yaml` is dumped.
    """
    out = Path(cfg.output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out

    if is_resumable(out) and not cfg.fresh:
        manifest = read_manifest(out / "manifest.json")
        return out, manifest

    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    dump_experiment_config(cfg, out / "config.yaml")

    manifest = Manifest(
        experiment_id=cfg.experiment_id,
        git_sha=current_git_sha(),
        seed=cfg.seed,
        start_iso=datetime.now(UTC).isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        resume_from=cfg.resume_from_manifest,
        fresh=cfg.fresh,
    )
    write_manifest(out / "manifest.json", manifest)
    return out, manifest
