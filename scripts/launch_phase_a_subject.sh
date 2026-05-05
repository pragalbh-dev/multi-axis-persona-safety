#!/usr/bin/env bash
# Launch a Phase A pipeline for one of the 3 multi-subject runs (may_3_directive).
# Usage: launch_phase_a_subject.sh {qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on}
#
# Skips:
#   step 6  — steered runs are Phase B work; deferred per may_3_directive
#   step 8  — GPT-5.5 cross-judge dropped per 2026-04-30 amendment
#   step 10 — per-subject figures deferred to Phase F (cross-subject panels)
#
# Single-GPU host (1× RTX PRO 6000 96 GB): runs sequentially. Verify GPU is free
# before launching; this script does NOT check.
set -euo pipefail
SUBJECT="${1:-}"
case "$SUBJECT" in
    qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on) ;;
    *) echo "usage: $0 {qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on}" >&2; exit 2 ;;
esac
cd /home/ubuntu/research/multi-axis-persona-safety
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase_a_${SUBJECT}_${TS}.log"
nohup setsid env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run python -m src.experiments.plan_b \
    --subject "$SUBJECT" \
    --skip 6 8 10 \
  > "$LOG" 2>&1 &
PID=$!
disown $PID
echo "subject: $SUBJECT"
echo "PID: $PID"
echo "LOG: $LOG"
