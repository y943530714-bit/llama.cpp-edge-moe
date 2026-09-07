#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: 03_baseline_bench.sh --model MODEL [options]

Options:
  --llama-cli PATH       llama-cli executable (default: ./build/bin/llama-cli)
  --prompt TEXT          prompt to evaluate
  --ctx-size N           context size (default: 8192)
  --predict N            generated tokens (default: 128)
  --threads N            CPU threads (default: 8)
  --gpu-layers N         GPU layers (default: auto)
  --out-dir PATH         output directory (default: ./m0-results)
  --no-warmup            disable the llama.cpp warmup run
EOF
}

llama_cli="./build/bin/llama-cli"
model=""
prompt="Measure the baseline latency and throughput of this model."
ctx_size=8192
predict=128
threads=8
gpu_layers="auto"
out_dir="./m0-results"
no_warmup=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-cli) llama_cli="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --prompt) prompt="$2"; shift 2 ;;
        --ctx-size) ctx_size="$2"; shift 2 ;;
        --predict) predict="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --gpu-layers) gpu_layers="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --no-warmup) no_warmup=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$model" ]]; then
    echo "--model is required" >&2
    exit 2
fi
if [[ ! -x "$llama_cli" ]]; then
    echo "llama-cli is not executable: $llama_cli" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

mkdir -p -- "$out_dir"
if [[ -n "${EDGE_MOE_BASELINE_MODES:-}" ]]; then
    IFS=',' read -r -a modes <<< "$EDGE_MOE_BASELINE_MODES"
else
    modes=(mmap none)
fi

time_tool=(/usr/bin/time)
if [[ ! -x "${time_tool[0]}" ]]; then
    time_tool=(time)
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
    time_args=(-l)
else
    time_args=(-v)
fi

print_system_state() {
    echo "--- system state ---"
    uname -a || true
    if command -v sysctl >/dev/null 2>&1; then
        sysctl vm.swapusage 2>/dev/null || true
    fi
    if command -v vm_stat >/dev/null 2>&1; then
        vm_stat 2>/dev/null | head -20 || true
    fi
}

for mode in "${modes[@]}"; do
    log_file="$out_dir/baseline-${mode}.log"
    echo "=== load mode: $mode ==="
    print_system_state
    command=("$llama_cli" -m "$model" --load-mode "$mode" -ngl "$gpu_layers" -c "$ctx_size" -n "$predict" -t "$threads" -p "$prompt" --single-turn)
    if [[ "$no_warmup" -eq 1 ]]; then
        command+=(--no-warmup)
    fi
    echo "command: ${command[*]}"
    "${time_tool[@]}" "${time_args[@]}" "${command[@]}" 2>&1 | tee "$log_file"
    print_system_state
    echo "log: $log_file"
done
