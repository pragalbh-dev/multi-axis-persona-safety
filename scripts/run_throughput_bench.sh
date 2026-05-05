#!/usr/bin/env bash
# Throughput rough-gist benchmark.
#
# 4 cells: {HF, SGLang} × {unsteered, capped}, on Gemma 2 27B.
# 100 prompts × 256 max_new_tokens × batch_size=32, bf16, T=0.
#
# Note: SGLang capped runs with --disable-piecewise-cuda-graph (the spike
# host doesn't support the JIT path; full piecewise graphs would need a
# 12-min compile per launch). HF baseline has no equivalent overhead.

set -euo pipefail
REPO=/home/ubuntu/research/multi-axis-persona-safety
cd "$REPO"

PORT=30000
MODEL=google/gemma-2-27b-it

echo "==[1/4] HF unsteered=="
source .venv/bin/activate
python scripts/bench_sglang_vs_hf.py --backend hf --model-path "$MODEL" --condition unsteered --layer 22
deactivate

echo "==[2/4] HF capped=="
source .venv/bin/activate
python scripts/bench_sglang_vs_hf.py --backend hf --model-path "$MODEL" --condition capped --layer 22
deactivate

# Free GPU; ensure clean start for SGLang
sleep 5

echo "==[3/4] SGLang unsteered=="
source .venv-sglang/bin/activate
LOG=logs/bench_sgl_unsteered_$(date +%Y%m%d_%H%M%S).log
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=False \
  nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" --tp 1 --dtype bfloat16 --port $PORT \
    --mem-fraction-static 0.85 --attention-backend triton \
    > "$LOG" 2>&1 &
SPID=$!
disown $SPID
for i in {1..120}; do
  grep -q "Application startup complete" "$LOG" 2>/dev/null && break
  grep -q -E "Traceback|RuntimeError|FAILED" "$LOG" 2>/dev/null && { echo "ERR"; tail -10 "$LOG"; exit 3; }
  sleep 5
done
python scripts/bench_sglang_vs_hf.py --backend sglang --model-path "$MODEL" --condition unsteered --port $PORT
kill -9 $SPID 2>/dev/null || true
pgrep -f sglang.launch_server | xargs -r kill -9 2>/dev/null || true
sleep 8

echo "==[4/4] SGLang capped=="
LOG=logs/bench_sgl_capped_$(date +%Y%m%d_%H%M%S).log
CAPPING_JSON=$(jq -c '.capping' "$REPO/results/sglang_spike/gemma2/forward_hooks_specs.json")
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=False \
  nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" --tp 1 --dtype bfloat16 --port $PORT \
    --mem-fraction-static 0.85 --attention-backend triton \
    --disable-piecewise-cuda-graph \
    --forward-hooks "$CAPPING_JSON" \
    > "$LOG" 2>&1 &
SPID=$!
disown $SPID
for i in {1..120}; do
  grep -q "Application startup complete" "$LOG" 2>/dev/null && break
  grep -q -E "Traceback|RuntimeError|FAILED" "$LOG" 2>/dev/null && { echo "ERR"; tail -10 "$LOG"; exit 3; }
  sleep 5
done
python scripts/bench_sglang_vs_hf.py --backend sglang --model-path "$MODEL" --condition capped --port $PORT
kill -9 $SPID 2>/dev/null || true
pgrep -f sglang.launch_server | xargs -r kill -9 2>/dev/null || true
deactivate

echo "DONE. See $REPO/results/sglang_spike/throughput.json"
cat "$REPO/results/sglang_spike/throughput.json" | jq -r '.rows[] | "\(.backend) \(.condition): \(.tokens_per_sec) tps"'
