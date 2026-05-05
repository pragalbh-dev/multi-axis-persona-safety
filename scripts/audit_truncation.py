"""Quick truncation audit for a Phase A subject.

For each rollout parquet (role_rollouts + per-condition rollouts), report
percentile token lengths and the fraction at exactly max_new_tokens (truncated).

Usage: audit_truncation.py <subject_id> [<max_new_tokens_override>]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: audit_truncation.py <subject_id> [max_new_tokens]", file=sys.stderr)
        return 2
    subject = sys.argv[1]
    out_dir = Path("results/phase_a") / subject
    if not out_dir.is_dir():
        print(f"FAIL: {out_dir} does not exist", file=sys.stderr)
        return 1

    cfg_path = out_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    max_new = int(sys.argv[2]) if len(sys.argv) >= 3 else int(cfg["max_new_tokens"])

    parquets: list[tuple[str, Path]] = []
    rr = out_dir / "role_rollouts.parquet"
    if rr.exists():
        parquets.append(("role_rollouts", rr))
    rollouts_dir = out_dir / "rollouts"
    if rollouts_dir.is_dir():
        for p in sorted(rollouts_dir.glob("*.parquet")):
            if p.name.startswith("_"):
                continue
            parquets.append((p.stem, p))

    print(f"\n=== truncation audit: {subject} (max_new_tokens={max_new}) ===")
    if not parquets:
        print("(no parquets found yet)")
        return 0

    any_high = False
    for name, p in parquets:
        df = pd.read_parquet(p)
        if "response_tokens" not in df.columns:
            print(f"  {name:30s}: no response_tokens column")
            continue
        t = df["response_tokens"]
        truncated = int((t >= max_new).sum())
        pct = 100 * truncated / max(1, len(t))
        flag = ""
        if name != "role_rollouts" and pct > 50:
            flag = "  ⚠ HIGH (>50% on a non-extraction condition affects judge labels)"
            any_high = True
        elif name == "role_rollouts" and pct > 95:
            flag = "  ⚠ extreme (PCA still works but at narrowed window)"
        print(
            f"  {name:30s}: n={len(t):4d}  p50={int(t.median()):4d}  p95={int(t.quantile(0.95)):4d}  "
            f"p99={int(t.quantile(0.99)):4d}  max={int(t.max()):4d}  truncated={truncated:4d} ({pct:5.1f}%){flag}"
        )

    print()
    if any_high:
        print("WARNING: at least one non-extraction condition has >50% truncation.")
        print("Consider re-running step 5 with a higher max_new_tokens.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
