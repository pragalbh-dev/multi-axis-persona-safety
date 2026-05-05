#!/usr/bin/env bash
# Phase B chain launcher. Watches for each subject's lambda_sweep.parquet
# (the unambiguous completion marker) and auto-launches the next subject
# after the sanity gate passes.
#
# Usage: chain_phase_b_subjects.sh subject1 [subject2 ...]
# Example: chain_phase_b_subjects.sh qwen_3_32b gemma_4_31b_thinking_off gemma_4_31b_thinking_on
#
# - If a subject's lambda_sweep.parquet already exists: skipped.
# - If a subject's process is already running: chain just waits on the file.
# - If a subject's process disappears without writing the file: chain halts.
# - Sanity gate runs after each completion; halts loudly on failure.
# - On full success, writes results/phase_b/.chain_complete sentinel.
set -euo pipefail
cd /home/ubuntu/research/multi-axis-persona-safety
mkdir -p logs results/phase_b

CHAIN_LOG="logs/phase_b_chain_$(date -u +%Y%m%d_%H%M%S).log"
SENTINEL="results/phase_b/.chain_complete"
rm -f "$SENTINEL"

ts() { date -u +%H:%M:%SZ; }
log() { echo "[$(ts) chain_b] $*" | tee -a "$CHAIN_LOG"; }

log "starting Phase B chain over: $*"
log "log file: $CHAIN_LOG"

for SUBJECT in "$@"; do
    DONE_FILE="results/phase_b/${SUBJECT}/lambda_sweep.parquet"

    if [[ -f "$DONE_FILE" ]]; then
        log "$SUBJECT: lambda_sweep.parquet already exists, skipping launch"
    else
        if pgrep -f "phase_b.*--subject ${SUBJECT}\b" > /dev/null; then
            log "$SUBJECT: already running (pgrep hit), waiting on lambda_sweep.parquet"
        else
            log "$SUBJECT: launching via scripts/launch_phase_b_subject.sh"
            scripts/launch_phase_b_subject.sh "$SUBJECT" 2>&1 | tee -a "$CHAIN_LOG"
            sleep 20
        fi

        # Wait. Re-check process every 60s; bail if it dies without the file.
        while [[ ! -f "$DONE_FILE" ]]; do
            if ! pgrep -f "phase_b.*--subject ${SUBJECT}\b" > /dev/null; then
                log "ERROR: $SUBJECT process not running and lambda_sweep.parquet missing — chain halted"
                exit 1
            fi
            sleep 60
        done
        log "$SUBJECT: lambda_sweep.parquet landed — running sanity gate"
    fi

    # Sanity gate: catches silent failures (degenerate v_harm, AA-cap broke
    # nothing, no axis picked, etc.). Halts loudly so the human investigates
    # before the next subject's GPU time is sunk.
    if ! uv run python scripts/sanity_check_phase_b.py "$SUBJECT" 2>&1 | tee -a "$CHAIN_LOG"; then
        log "ERROR: $SUBJECT FAILED SANITY GATE — chain halted; investigate before continuing"
        exit 2
    fi
    log "$SUBJECT: COMPLETE + sanity gate passed"
done

touch "$SENTINEL"
log "PHASE B CHAIN COMPLETE — all subjects done + all sanity gates passed, sentinel: $SENTINEL"
