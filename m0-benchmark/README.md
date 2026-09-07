# Edge MoE M0 benchmark

These scripts collect the first measurements needed by the edge MoE plan:

- SSD sequential and random read throughput
- the process `mlock` ceiling
- llama.cpp baseline timings and memory information
- routed expert IDs and weights from real model execution
- offline route coverage and cache simulations

Build `llama-cli` first. From the repository root:

```bash
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

Measure the storage and memory limits on the target machine:

```bash
./m0-benchmark/01_ssd_bench.sh /path/to/fast/storage 1
python3 m0-benchmark/02_mlock_probe.py --max-gb 8
```

Run a baseline. The script runs the modes in `EDGE_MOE_BASELINE_MODES`, which
defaults to `mmap,none`:

```bash
./m0-benchmark/03_baseline_bench.sh \
  --model /path/to/model.gguf \
  --llama-cli ./build/bin/llama-cli \
  --ctx-size 8192 \
  --predict 128
```

Collect a real route trace. The option is disabled unless explicitly set and
does not change the model graph or its routing decisions. Warmup is disabled
by default so the trace only contains the requested prompt and generation;
pass `--warmup` when warmup traffic is part of the experiment. Tracing adds
tensor copies and scheduler observation points, so use it for route data rather
than latency measurements:

```bash
python3 m0-benchmark/04_expert_trace.py \
  --model /path/to/model.gguf \
  --llama-cli ./build/bin/llama-cli \
  --prompt "Explain how a SSD-backed expert cache should be evaluated." \
  --trace traces/qwen35-moe.jsonl \
  --predict 32 \
  --ctx-size 8192
```

Inspect the trace without third-party Python packages:

```bash
python3 m0-benchmark/05_expert_cache_sim.py traces/qwen35-moe.jsonl --stage all
```

`--stage all` keeps the prefill-to-decode cache handoff. `--stage decode`
starts from an empty cache and is useful for isolating decode behavior.

The simulator can compare a no-prefetch baseline with an oracle upper bound or
a static frequency prefetch policy. For the current M4 estimate, start with a
1.5-3.0 GB arena and a 20-event prefetch latency sweep:

```bash
python3 m0-benchmark/05_expert_cache_sim.py traces/qwen35-moe.jsonl \
  --stage decode --capacity-gb 1.5 --prefetch none

python3 m0-benchmark/05_expert_cache_sim.py traces/qwen35-moe.jsonl \
  --stage decode --capacity-gb 1.5 --prefetch oracle \
  --prefetch-horizon 20 --prefetch-latency 20
```

Use synthetic traces to validate the simulator itself:

```bash
python3 m0-benchmark/06_gen_synthetic_trace.py \
  --output /tmp/edge-moe-synthetic.jsonl \
  --layers 40 --experts 256 --topk 8 --decode-tokens 256
python3 m0-benchmark/05_expert_cache_sim.py /tmp/edge-moe-synthetic.jsonl \
  --stage decode --capacity-gb 3 --prefetch frequency
```

## Trace format

The trace is JSON Lines. The first line is a header. Each route record stores
expert IDs and weights in token-major order:

```json
{"schema_version":1,"event":"route","layer":0,"stage":"decode","n_tokens":1,"n_experts_used":8,"weights_normalized":true,"expert_ids":[[1,4,7,9,12,18,22,31]],"weights":[[0.2,0.18,0.15,0.12,0.1,0.09,0.08,0.08]]}
```

`stage` is inferred from the graph batch: a route with one token is marked as
`decode`, otherwise it is marked as `prefill`. Raw weights are retained when a
model does not expose a normalized routing tensor; the record then sets
`weights_normalized` to `false`.
