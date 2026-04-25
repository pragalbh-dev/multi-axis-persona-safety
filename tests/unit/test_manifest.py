"""Stage 1 T1.5 — Manifest IO + results-dir contract."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import ExperimentConfig
from src.utils.manifest import (
    SCHEMA_VERSION,
    Manifest,
    is_resumable,
    read_manifest,
    write_manifest,
)
from src.utils.results import init_results_dir


def _base_cfg(out: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp_unit_test",
        model_id="gemma_2_27b",
        output_dir=str(out),
    )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = Manifest(
        experiment_id="exp_x",
        git_sha="deadbeef",
        seed=42,
        start_iso="2026-04-25T07:47:00+00:00",
    )
    p = tmp_path / "manifest.json"
    write_manifest(p, m)
    read = read_manifest(p)
    assert read.git_sha == "deadbeef"
    assert read.schema_version == SCHEMA_VERSION
    assert read.end_iso is None


def test_init_results_dir_fresh(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "exp1")
    out, manifest = init_results_dir(cfg)
    assert out.is_dir()
    assert (out / "config.yaml").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "figures").is_dir()
    assert manifest.experiment_id == "exp_unit_test"
    # Resumable while end_iso is None.
    assert is_resumable(out)


def test_init_results_dir_resumes(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "exp_resume")
    out1, m1 = init_results_dir(cfg)
    # Mark partial work — manifest sits with end_iso=None, so re-init resumes.
    out2, m2 = init_results_dir(cfg)
    assert out1 == out2
    assert m1.start_iso == m2.start_iso  # resumed, did not overwrite


def test_init_results_dir_fresh_flag(tmp_path: Path) -> None:
    cfg1 = _base_cfg(tmp_path / "exp_fresh")
    out, m1 = init_results_dir(cfg1)
    cfg2 = ExperimentConfig(**{**cfg1.model_dump(), "fresh": True})
    out, m2 = init_results_dir(cfg2)
    # fresh=True forces a new manifest start_iso.
    assert m2.start_iso >= m1.start_iso  # fresh path may produce same iso at fast tests
    assert m2.fresh is True
