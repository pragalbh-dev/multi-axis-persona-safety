"""Tests for run_in_subprocess.

Uses a tiny no-GPU echo work-module (in tests/) to verify the subprocess
wrapper correctly serializes args, captures output JSON, propagates errors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.utils.model_runner import WorkModuleError, run_in_subprocess

# An "echo" work-module that lives in the tests/ tree.
TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TESTS_DIR))


def test_run_in_subprocess_echo_roundtrip() -> None:
    """Echo work-module: receives args, writes them back into output JSON."""
    args = {"hello": "world", "n": 42, "list": [1, 2, 3]}
    out = run_in_subprocess(
        "tests.fixtures.echo_work_module",
        args,
        timeout_seconds=30,
    )
    assert out["status"] == "ok"
    assert out["echoed_args"] == args
    assert "elapsed_seconds" in out


def test_run_in_subprocess_propagates_failure() -> None:
    """Child process exits non-zero on failure; wrapper raises WorkModuleError."""
    with pytest.raises(WorkModuleError):
        run_in_subprocess(
            "tests.fixtures.echo_work_module",
            {"force_failure": True},
            timeout_seconds=30,
        )


def test_run_in_subprocess_writes_to_explicit_output_path(tmp_path: Path) -> None:
    """When `output_path` is provided, the result is persisted there for inspection."""
    out_path = tmp_path / "result.json"
    args = {"hello": "explicit"}
    out = run_in_subprocess(
        "tests.fixtures.echo_work_module",
        args,
        output_path=out_path,
        timeout_seconds=30,
    )
    assert out_path.exists()
    assert out["echoed_args"] == args
    on_disk = json.loads(out_path.read_text())
    assert on_disk["echoed_args"] == args
