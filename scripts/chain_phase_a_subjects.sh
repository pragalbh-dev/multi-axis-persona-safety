#!/usr/bin/env bash
# Phase A chain launcher. Watches for each subject's metrics.json (the
# unambiguous completion marker) and auto-launches the next subject.
# Replaces the fragile "tail -F | grep PLAN B COMPLETE" pattern that left
# the GPU idle for 12 hr on 2026-04-30/05-01.
#
# Usage: chain_phase_a_subjects.sh subject1 [subject2 ...]
# Example: chain_phase_a_subjects.sh gemma_4_31b_thinking_off gemma_4_31b_thinking_on
#
# - If a subject's metrics.json already exists: skipped.
# - If a subject's process is already running: chain just waits on metrics.json.
# - If a subject's process disappears without writing metrics.json: chain halts.
# - On full success, writes results/phase_a/.chain_complete sentinel.
# - Logs progress to logs/phase_a_chain.log (timestamped + tee'd to stdout).
set -euo pipefail
cd /home/ubuntu/research/multi-axis-persona-safety
mkdir -p logs results/phase_a

CHAIN_LOG="logs/phase_a_chain_$(date -u +%Y%m%d_%H%M%S).log"
SENTINEL="results/phase_a/.chain_complete"
rm -f "$SENTINEL"

ts() { date -u +%H:%M:%SZ; }
log() { echo "[$(ts) chain] $*" | tee -a "$CHAIN_LOG"; }

log "starting chain over: $*"
log "log file: $CHAIN_LOG"

for SUBJECT in "$@"; do
    METRICS="results/phase_a/${SUBJECT}/metrics.json"

    if [[ -f "$METRICS" ]]; then
        log "$SUBJECT: metrics.json already exists, skipping launch"
        continue
    fi

    # Launch only if not already running.
    if pgrep -f "plan_b.*--subject ${SUBJECT}\b" > /dev/null; then
        log "$SUBJECT: already running (pgrep hit), waiting on metrics.json"
    else
        log "$SUBJECT: launching via scripts/launch_phase_a_subject.sh"
        scripts/launch_phase_a_subject.sh "$SUBJECT" 2>&1 | tee -a "$CHAIN_LOG"
        sleep 20
    fi

    # Wait. Re-check process every 60s; bail if it dies without metrics.json.
    while [[ ! -f "$METRICS" ]]; do
        if ! pgrep -f "plan_b.*--subject ${SUBJECT}\b" > /dev/null; then
            log "ERROR: $SUBJECT process not running and metrics.json missing — chain halted"
            exit 1
        fi
        sleep 60
    done
    log "$SUBJECT: metrics.json landed — running sanity gate"

    # Sanity gate: programmatic plausibility checks BEFORE launching next subject.
    # Catches silent failures that produce a metrics.json but with garbage values
    # (empty responses, judge meltdown, broken extraction, low cos_sim, no LASSO
    # selection, etc.). Halts the chain loudly so the human can investigate
    # before the next subject's GPU time is sunk.
    if ! uv run python scripts/sanity_check_phase_a.py "$SUBJECT" 2>&1 | tee -a "$CHAIN_LOG"; then
        log "ERROR: $SUBJECT FAILED SANITY GATE — chain halted; investigate before continuing"
        exit 2
    fi
    log "$SUBJECT: COMPLETE + sanity gate passed"
done

touch "$SENTINEL"
log "PHASE A CHAIN COMPLETE — all subjects done + all sanity gates passed, sentinel: $SENTINEL"
