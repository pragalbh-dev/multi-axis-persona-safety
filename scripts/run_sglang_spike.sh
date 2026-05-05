#!/usr/bin/env bash
# End-to-end driver for the SGLang spike.
#
# Prerequisites:
#   - .venv (main project) and .venv-sglang both built
#   - cuda-toolkit-12-9 installed (for SGLang JIT compile of sm_120 kernels)
#   - Gemma 2 27B + Gemma 4 31B weights cached locally
#   - `python -m tests.integration.sglang_hooks_smoke --phase setup ...` already run
#
# Usage:
#   bash scripts/run_sglang_spike.sh gemma2   # Gemma 2 27B end-to-end
#   bash scripts/run_sglang_spike.sh gemma4   # Gemma 4 31B end-to-end
#   bash scripts/run_sglang_spike.sh compare  # just run the comparison phase
#
# Each subject takes ~50 min (HF reference all patterns + SGLang per-pattern relaunch).

set -euo pipefail

REPO=/home/ubuntu/research/multi-axis-persona-safety
RESULTS=$REPO/results/sglang_spike
LOGDIR=$REPO/logs

# Single source of truth for CUDA env passed to SGLang processes.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_ENABLE_JIT_DEEPGEMM=False

PATTERNS=(unsteered addition_pos addition_neg capping cap_and_steer multi_axis_cap)
SUBJECT=${1:-gemma2}

case "$SUBJECT" in
  gemma2)
    MODEL_PATH=google/gemma-2-27b-it
    HIDDEN_SIZE=4608
    LAYER=22
    HOOK_GLOB_PREFIX="model.layers"
    ;;
  gemma4)
    MODEL_PATH=google/gemma-4-31B-it
    HIDDEN_SIZE=5376  # placeholder, will be read from config
    LAYER=24          # placeholder; tune to mid-depth
    HOOK_GLOB_PREFIX="model.language_model.layers"
    ;;
  compare)
    cd "$REPO" && source .venv-sglang/bin/activate \
      && python -m tests.integration.sglang_hooks_smoke --phase compare
    exit 0
    ;;
  *)
    echo "unknown subject: $SUBJECT" >&2; exit 2 ;;
esac

cd "$REPO"

echo "==[1/3] HF reference, all patterns, $SUBJECT=="
source .venv/bin/activate
for pat in "${PATTERNS[@]}"; do
  echo "--- HF: $pat ---"
  python -m tests.integration.sglang_hooks_smoke \
    --phase hf --pattern "$pat" --model-path "$MODEL_PATH" --layer "$LAYER"
done
deactivate

echo "==[2/3] SGLang, per pattern (server relaunched each time)=="
source .venv-sglang/bin/activate
for pat in "${PATTERNS[@]}"; do
  echo "--- SGLang: $pat ---"
  HOOKS_JSON=$(python -c "import json; d=json.load(open('$RESULTS/forward_hooks_specs.json')); print(json.dumps(d['$pat']))")
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG=$LOGDIR/sglang_${SUBJECT}_${pat}_${STAMP}.log
  echo "[server] launching with hooks=$pat, log=$LOG"

  CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=False \
    nohup setsid python -m sglang.launch_server \
      --model-path "$MODEL_PATH" --tp 1 --dtype bfloat16 --port 30000 \
      --mem-fraction-static 0.85 \
      --attention-backend triton \
      --forward-hooks "$HOOKS_JSON" \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
  disown $SERVER_PID

  # Wait for server ready (max 8 min)
  for i in {1..96}; do
    if grep -q -E "Application startup complete|fired up|Uvicorn running" "$LOG" 2>/dev/null; then
      echo "[server] ready after ${i}*5s"; break
    fi
    if grep -q -E "RuntimeError|Traceback|FAILED|status 127" "$LOG" 2>/dev/null; then
      echo "[server] failed — see $LOG" >&2; tail -20 "$LOG" >&2; exit 3
    fi
    sleep 5
  done

  python -m tests.integration.sglang_hooks_smoke \
    --phase sglang --pattern "$pat" --model-path "$MODEL_PATH" --port 30000

  # Tear down server
  kill $SERVER_PID 2>/dev/null || true
  pgrep -f "sglang.launch_server" | xargs -r kill 2>/dev/null || true
  sleep 5
done
deactivate

echo "==[3/3] compare=="
source .venv-sglang/bin/activate
python -m tests.integration.sglang_hooks_smoke --phase compare
deactivate

echo "DONE. See $RESULTS/equivalence.json"
