"""Experiment manifest — a small JSON sidecar in every results dir.

Locked by Stage 1 / T1.5. Every results/exp{N}_{name}/ contains:
- config.yaml      ← `dump_experiment_config(cfg, ...)` output
- manifest.json    ← this file (write at start, update at end)
- metrics.json     ← aggregate numbers (written by analysis)
- details.parquet  ← per-prompt rows
- figures/         ← matplotlib + plotly figures

The manifest exists so `--fresh`-vs-resume logic and downstream agents can
inspect what was run without re-loading every artifact.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Manifest:
    experiment_id: str
    git_sha: str
    seed: int
    start_iso: str
    end_iso: str | None = None
    schema_version: int = SCHEMA_VERSION
    artifacts: dict[str, str] = field(default_factory=dict)
    resume_from: str | None = None
    fresh: bool = False

    def mark_done(self) -> None:
        self.end_iso = datetime.now(UTC).isoformat(timespec="seconds")

    def add_artifact(self, key: str, path: str | Path) -> None:
        self.artifacts[key] = str(path)


def current_git_sha() -> str:
    """Resolve HEAD SHA; return 'unknown' if not in a git work tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=False)


def read_manifest(path: str | Path) -> Manifest:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest(**data)


def is_resumable(results_dir: str | Path) -> bool:
    """A results dir is resumable iff manifest.json exists and end_iso is null."""
    p = Path(results_dir) / "manifest.json"
    if not p.is_file():
        return False
    try:
        m = read_manifest(p)
    except (json.JSONDecodeError, TypeError):
        return False
    return m.end_iso is None
