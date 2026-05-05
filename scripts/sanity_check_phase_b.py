"""Per-subject sanity gate after Phase B completes.

Checks the orchestrator's outputs are not just present but plausible:
  1. Required artifacts: lambda_sweep.parquet, lambda_pareto.json, headline.json,
     harm_direction.json, full_judged.parquet, mini_judged.parquet
  2. Judge parse rate ≥ 95% on the full pass
  3. AA-cap baseline reproduces some harm reduction (>0 pp; paper expects ~30+ pp)
  4. lambda_pareto non-empty (i.e., at least one axis chose a λ)
  5. headline.json: at least one PC attack reports recovery_pp_vs_aa_cap > 0 or
     coherence_rate < 0.95 (i.e., the attack DID something)
  6. v_harm_norm_pre is non-degenerate (>1.0; means we have signal)

Exits 0 if all pass, 1 if any fail. Used by scripts/chain_phase_b_subjects.sh
to halt the chain before launching the next subject.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = [
    "lambda_sweep.parquet",
    "lambda_pareto.json",
    "headline.json",
    "harm_direction.json",
    "rollouts/mini_judged.parquet",
    "rollouts/full_judged.parquet",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sanity_check_phase_b.py <subject_id>", file=sys.stderr)
        return 2
    subject = sys.argv[1]
    root = Path("results/phase_b") / subject
    if not root.is_dir():
        print(f"FAIL: results dir does not exist: {root}")
        return 1

    failures: list[str] = []

    # Gate 1: required artifacts
    for rel in REQUIRED:
        if not (root / rel).exists():
            failures.append(f"missing artifact: {rel}")
    if failures:
        for f in failures:
            print(f"FAIL [{subject}]: {f}")
        return 1

    headline = json.loads((root / "headline.json").read_text())
    harm_dir = json.loads((root / "harm_direction.json").read_text())
    pareto = json.loads((root / "lambda_pareto.json").read_text())

    # Gate 2: judge parse rate (re-derived from step6 marker if available)
    s6 = root / ".step6.done"
    if s6.exists():
        s6d = json.loads(s6.read_text())
        n_parsed = int(s6d.get("n_parsed", 0))
        n_rows = int(s6d.get("n_rows", 1))
        if n_rows > 0 and n_parsed / n_rows < 0.95:
            failures.append(f"judge parse rate {n_parsed}/{n_rows} = {n_parsed/n_rows:.3f} < 0.95")

    # Gate 3: AA-cap reduction non-zero
    cap_red = headline.get("aa_cap_reduction_pp")
    if cap_red is None:
        failures.append("aa_cap_reduction_pp missing — full_aa_capped_only didn't run?")
    elif cap_red <= 0.0:
        failures.append(f"aa_cap_reduction_pp = {cap_red:.2f} ≤ 0 — cap didn't reduce harm")

    # Gate 4: pareto non-empty
    chosen = pareto.get("chosen_lambda_per_axis", {}) or {}
    if not chosen:
        failures.append("lambda_pareto.chosen_lambda_per_axis empty")

    # Gate 5: at least one PC attack did SOMETHING (recovered harm or hurt coherence)
    per_attack = headline.get("per_attack_full", {}) or {}
    any_signal = False
    for cond, v in per_attack.items():
        if not cond.startswith("full_aa_capped_signmatched_pc"):
            continue
        rec = v.get("recovery_pp_vs_aa_cap")
        coh = v.get("coherence_rate", 1.0)
        if (rec is not None and rec > 0.0) or coh < 0.95:
            any_signal = True
            break
    if per_attack and not any_signal:
        failures.append("no PC attack showed harm recovery > 0 or coherence loss > 5%")

    # Gate 6: v_harm signal
    vhn = float(harm_dir.get("v_harm_norm_pre", 0.0))
    if vhn < 1.0:
        failures.append(f"v_harm_norm_pre = {vhn:.3f} < 1.0 — degenerate harm signal")

    if failures:
        print(f"\n=== SANITY FAIL [{subject}] ===")
        for f in failures:
            print(f"  - {f}")
        print(f"\nArtifacts at {root}/")
        return 1

    # Success summary
    print(
        f"SANITY OK [{subject}]: "
        f"baseline={headline['baseline_harm_rate']:.3f}  "
        f"aa_cap_only={headline['aa_cap_only_harm_rate']:.3f}  "
        f"reduction={cap_red:+.2f}pp  "
        f"cos(v_harm,AA)={harm_dir['cos_sim_v_harm_aa']:+.3f}  "
        f"|v_harm|={vhn:.3f}  "
        f"axes_chosen={len(chosen)}"
    )
    for cond, v in per_attack.items():
        print(
            f"  {cond}: harm={v['harm_rate']:.3f} "
            f"coherence={v['coherence_rate']:.3f} "
            f"recovery={v['recovery_pp_vs_aa_cap']:+.2f}pp"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
