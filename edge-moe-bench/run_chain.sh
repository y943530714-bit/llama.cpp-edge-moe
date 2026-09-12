#!/bin/bash
# Chain the remaining edge MoE benchmark configs after the baseline finishes.
set -u
cd /d/workspace/EdgeMoe
PY="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
MODEL="D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DRAFT="D:\\workspace\\models\\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"
LOG=edge-moe-results/chain.log

echo "=== chain start $(date) ===" >> "$LOG"

# wait until the baseline driver has written its summary
for i in $(seq 1 240); do
  test -f edge-moe-results/logs/baseline_summary.json && break
  sleep 30
done
if ! test -f edge-moe-results/logs/baseline_summary.json; then
  echo "baseline never finished" >> "$LOG"; exit 1
fi
echo "=== baseline done $(date) ===" >> "$LOG"

# opt1: expert skipping k1=4 k2=16
"$PY" edge-moe-bench/run_bench.py --model "$MODEL" --config-name opt1_skip \
  --predict 32 --ws-cap-gb 6.5 \
  --extra "--temp 0 --moe-skip-k1 4 --moe-skip-k2 16" \
  --out edge-moe-results/bench.csv > edge-moe-results/opt1_run.log 2>&1
echo "opt1 exit $? $(date)" >> "$LOG"

# rebuild with the relaxed-verify server wiring before the spec runs
CMAKE="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
"$CMAKE" --build build-win --config Release --parallel 2 --target llama-cli llama-server \
  > edge-moe-results/rebuild.log 2>&1
BUILD=$?
echo "rebuild exit $BUILD $(date)" >> "$LOG"
if [ "$BUILD" != "0" ]; then
  echo "rebuild failed, stopping chain" >> "$LOG"
  grep -i "error" edge-moe-results/rebuild.log | head -5 >> "$LOG"
  exit 2
fi

# opt2: opt1 + DFlash draft (Q4_K_M) speculative decoding, draft length 7
"$PY" edge-moe-bench/run_bench.py --model "$MODEL" --config-name opt2_dflash \
  --predict 32 --ws-cap-gb 6.5 \
  --extra "--temp 0 --moe-skip-k1 4 --moe-skip-k2 16 -md $DRAFT --spec-type draft-dflash --spec-draft-n-max 7" \
  --out edge-moe-results/bench.csv > edge-moe-results/opt2_run.log 2>&1
echo "opt2 exit $? $(date)" >> "$LOG"

# opt3: opt1+2 + relaxed verify
"$PY" edge-moe-bench/run_bench.py --model "$MODEL" --config-name opt3_relaxed \
  --predict 32 --ws-cap-gb 6.5 \
  --extra "--temp 0 --moe-skip-k1 4 --moe-skip-k2 16 -md $DRAFT --spec-type draft-dflash --spec-draft-n-max 7 --spec-relaxed-verify" \
  --out edge-moe-results/bench.csv > edge-moe-results/opt3_run.log 2>&1
echo "opt3 exit $? $(date)" >> "$LOG"

# trace run for the expert hit-rate analysis (opt1 config, per-problem traces)
mkdir -p edge-moe-results/traces
"$PY" edge-moe-bench/run_bench.py --model "$MODEL" --config-name trace_skip \
  --predict 32 --ws-cap-gb 6.5 \
  --extra "--temp 0 --moe-skip-k1 4 --moe-skip-k2 16 --moe-trace edge-moe-results/traces/{task}.jsonl" \
  --out edge-moe-results/trace_runs.csv > edge-moe-results/trace_run.log 2>&1
echo "trace exit $? $(date)" >> "$LOG"

echo "=== chain end $(date) ===" >> "$LOG"
