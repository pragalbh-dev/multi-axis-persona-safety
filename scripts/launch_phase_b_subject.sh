#!/usr/bin/env bash
# Launch a Phase B (attack arm) pipeline for one of the 3 multi-subject runs.
# Usage: launch_phase_b_subject.sh {qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on}
#
# Phase B requires the steered backend per configs/subjects.yaml::<id>.steered_backend:
#   - qwen_3_32b → sglang  (run from .venv-sglang)
#   - gemma_4_31b_* → hf   (run from .venv)
#
# Single-GPU host (1× RTX PRO 6000 96 GB): runs sequentially. Verify GPU is free.
set -euo pipefail
SUBJECT="${1:-}"
case "$SUBJECT" in
    qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on) ;;
    *) echo "usage: $0 {qwen_3_32b|gemma_4_31b_thinking_off|gemma_4_31b_thinking_on}" >&2; exit 2 ;;
esac
cd /home/ubuntu/research/multi-axis-persona-safety

# Pick the right venv based on subject's steered_backend.
case "$SUBJECT" in
    qwen_3_32b)
        VENV=".venv-sglang"
        ;;
    gemma_4_31b_thinking_off|gemma_4_31b_thinking_on)
        VENV=".venv"
        ;;
esac
PYTHON="$PWD/$VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: expected python at $PYTHON not found" >&2
    exit 3
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase_b_${SUBJECT}_${TS}.log"
nohup setsid env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$PYTHON" -m src.experiments.phase_b \
    --subject "$SUBJECT" \
  > "$LOG" 2>&1 &
PID=$!
disown $PID
echo "subject: $SUBJECT"
echo "venv:    $VENV"
echo "PID:     $PID"
echo "LOG:     $LOG"
