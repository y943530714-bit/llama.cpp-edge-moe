# EdgeMoE-Llama M0 / PR-3 运行手册

本文对应 `edge_moe_design_v1.1.md` 的 M0–PR-3。当前代码只做测量、路由观测和 expert range 验证，不会改变默认 llama.cpp 推理路径。

## 1. 当前交付范围

已实现：

- `--moe-trace` 路由 trace（兼容 `--edge-moe-trace` 和 `--dump-expert-trace`）；
- SSD 顺序/随机读取测试；
- `mlock` 上限探测；
- baseline 推理与内存记录；
- 离线 expert cache simulator；
- `llama-edge-moe-layout`：检查 GGUF routed expert tensor 的维度、量化 block 对齐、文件边界和 expert slice offset/size；
- 可选读取 expert slice，并调用 `ggml_validate_row_data` 验证原始数据。
- `llama-edge-moe-slot-probe`：验证固定 resident slots、blocking load、logical-to-physical remap，以及 generic CPU `ggml_mul_mat_id` 的等价结果。
- resident-slot lifecycle 自测：在用时禁止驱逐/重载，显式驱逐后允许复用，并保持 generation 单调递增。

尚未实现：

- 默认模型推理路径中的 expert streaming 和 per-layer resident slots 接入；
- 异步 I/O、P0/P1/P2 调度和自动 cache eviction；
- Gate-First split graph、prefill double buffer 和 MTP prefetch；
- `.moepack` sidecar。
- resident-slot 原型尚未接入默认模型推理 graph，也没有隐式 eviction。

因此，本手册中的结果用于确认 M0/PR-3 前置条件，不代表已经开启低内存 MoE 推理优化。

## 2. 准备环境

需要：

- Windows、macOS 或 Linux；
- C++17 编译器和 CMake；
- 目标 backend 的依赖。M4 Mac mini 使用 Metal；
- 一个实际的 Qwen3.5/Qwen3.6-35B-A3B GGUF 模型；
- 足够的临时磁盘空间。SSD 测试会创建并删除临时文件。

以下命令均从仓库根目录执行。

## 3. 编译

M4/Metal：

```bash
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-cli llama-edge-moe-layout llama-edge-moe-slot-probe
```

CPU-only：

```bash
cmake -B build -DGGML_METAL=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-cli llama-edge-moe-layout llama-edge-moe-slot-probe
```

Windows/MSVC CPU-only：

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build build-win --config Release --parallel --target llama-cli llama-edge-moe-layout llama-edge-moe-slot-probe
```

检查工具是否生成：

```bash
test -x ./build/bin/llama-cli
test -x ./build/bin/llama-edge-moe-layout
test -x ./build/bin/llama-edge-moe-slot-probe
```

## 4. 推荐执行顺序

### 4.1 先做工具自测

不需要模型：

```bash
./build/bin/llama-edge-moe-layout --self-test
```

预期输出包含：

```text
self-test: 4 expert ranges map byte-for-byte
```

### 4.2 测量存储路径

把测试目录放在与模型相同的 SSD 或卷上：

```bash
./m0-benchmark/01_ssd_bench.sh /path/to/model-volume 1
```

记录 sequential 和 random 两行 MiB/s。该脚本只测存储路径，不代表 expert cache 的最终吞吐。

### 4.3 探测可锁定内存上限

```bash
python3 m0-benchmark/02_mlock_probe.py --max-gb 8
```

如果系统限制较低，不要把失败的 `mlock` 数值直接当作 expert arena budget。后续 runtime 仍需根据实际 backend/host 可用内存留出安全边界。

### 4.4 记录 baseline

```bash
./m0-benchmark/03_baseline_bench.sh \
  --model /path/to/model.gguf \
  --llama-cli ./build/bin/llama-cli \
  --ctx-size 8192 \
  --predict 128 \
  --out-dir ./m0-results
```

默认比较 `mmap` 和 `none` 两种 `--load-mode`。如需指定模式：

```bash
EDGE_MOE_BASELINE_MODES=mmap,none ./m0-benchmark/03_baseline_bench.sh \
  --model /path/to/model.gguf
```

保留每次运行的 stdout、stderr、TTFT/TPOT/tok/s、RSS 和 swap/memory pressure 信息。

### 4.5 采集真实 route trace

trace 会增加 tensor copy 和 callback 开销，只用于采集路由，不用于 latency benchmark。默认不包含 llama-cli warmup：

```bash
python3 m0-benchmark/04_expert_trace.py \
  --model /path/to/model.gguf \
  --llama-cli ./build/bin/llama-cli \
  --prompt "Explain how an SSD-backed expert cache should be evaluated." \
  --trace ./m0-results/qwen35-route.jsonl \
  --predict 128 \
  --ctx-size 8192 \
  --gpu-layers auto
```

如果要把 warmup 也纳入样本：

```bash
python3 m0-benchmark/04_expert_trace.py \
  --model /path/to/model.gguf \
  --prompt-file ./prompts/example.txt \
  --trace ./m0-results/qwen35-route-warmup.jsonl \
  --warmup \
  --force
```

确认 trace 至少包含一个 `event=route` 记录，并检查 `stage`、`layer`、`expert_ids` 和 `weights`。

### 4.6 验证 expert range layout（PR-2）

先只检查 metadata 和选中的 expert slice，不读取完整 routed expert tensor：

```bash
./build/bin/llama-edge-moe-layout \
  --model /path/to/model.gguf \
  --expert 0
```

再读取每个匹配 tensor 的 expert 0 slice，并做 raw row-data 校验：

```bash
./build/bin/llama-edge-moe-layout \
  --model /path/to/model.gguf \
  --expert 0 \
  --verify
```

如果后续要测试 direct-I/O 所需的对齐，可额外指定目标边界：

```bash
./build/bin/llama-edge-moe-layout \
  --model /path/to/model.gguf \
  --expert 0 \
  --alignment 4096 \
  --verify
```

split GGUF 模型：

```bash
./build/bin/llama-edge-moe-layout \
  --model /path/to/model-00001-of-00003.gguf \
  --split /path/to/model-00002-of-00003.gguf \
  --split /path/to/model-00003-of-00003.gguf \
  --expert 0 \
  --verify
```

只有在需要穷举读取时才使用全部 expert：

```bash
./build/bin/llama-edge-moe-layout \
  --model /path/to/model.gguf \
  --all-experts \
  --verify
```

重点查看最后一行：

```text
summary: matched_tensors=... expert_ranges=... verified_ranges=... errors=0
```

验收要求：

1. `matched_tensors` 大于 0；
2. 不使用 `--alignment` 时 `errors=0`；
3. 使用 `--verify` 时每个选中 slice 都输出 `row_data=ok`；
4. `expert_stride` 与同一 tensor 的每个 expert slice 大小一致；
5. 不出现 `outside the source file`、`not aligned to the ggml type block size` 或 `does not match contiguous expert slices`。

`--alignment` 是额外的 I/O 边界检查。如果模型 slice 不满足 4096 字节对齐，只说明当前模型布局不能直接满足该 direct-I/O 假设，不代表 GGUF 本身损坏。

### 4.7 验证 blocking resident-slot prototype（PR-3）

先运行不需要模型的 backend 等价性自测：

```bash
./build/bin/llama-edge-moe-slot-probe --self-test
```

预期输出包含：

```text
resident slots: logical 1->physical 0, logical 3->physical 1
backend remap: ggml_mul_mat_id max_abs_diff=0
resident lifecycle: refcount/eviction/generation=ok
```

再对真实 GGUF 的某一层加载选中的 expert range。该命令只读取指定的 expert slice，不进入默认 llama.cpp 推理路径。对 `--part gate_up`，工具优先使用融合的 `ffn_gate_up_exps` tensor；如果模型使用分离的 `ffn_gate_exps` 和 `ffn_up_exps`，则自动为二者建立独立 slot arena 并逐一验证：

```bash
./build/bin/llama-edge-moe-slot-probe \
  --model /path/to/model.gguf \
  --layer 0 \
  --part gate_up \
  --experts 0,1,3,7 \
  --slots 4
```

融合 tensor 的预期最后一行包含：

```text
summary: loaded=4 slots=4 errors=0
```

分离 gate/up tensor 会分别输出两行上述 summary，并以以下汇总结束：

```text
combined_summary: tensors=2 loaded=8 slots_per_tensor=4 errors=0
```

如果指定的 expert 数量大于 `--slots`，当前原型会因没有空闲 slot 而失败；这是刻意保留的显式 blocking 行为，后续 runtime 再接入 safe-point eviction。

### 4.8 运行离线 cache simulator

先模拟完整的 prefill-to-decode handoff：

```bash
python3 m0-benchmark/05_expert_cache_sim.py \
  ./m0-results/qwen35-route.jsonl \
  --stage all \
  --capacity-gb 1.5 \
  --prefetch none
```

再对比 oracle 和 frequency 预取：

```bash
python3 m0-benchmark/05_expert_cache_sim.py \
  ./m0-results/qwen35-route.jsonl \
  --stage decode \
  --capacity-gb 1.5 \
  --prefetch oracle \
  --prefetch-horizon 20 \
  --prefetch-latency 20

python3 m0-benchmark/05_expert_cache_sim.py \
  ./m0-results/qwen35-route.jsonl \
  --stage decode \
  --capacity-gb 1.5 \
  --prefetch frequency \
  --prefetch-horizon 20
```

合成 trace 只用于验证 simulator：

```bash
python3 m0-benchmark/06_gen_synthetic_trace.py \
  --output /tmp/edge-moe-synthetic.jsonl \
  --layers 40 \
  --experts 256 \
  --topk 8 \
  --decode-tokens 256
python3 m0-benchmark/05_expert_cache_sim.py \
  /tmp/edge-moe-synthetic.jsonl \
  --stage decode \
  --capacity-gb 3 \
  --prefetch frequency
```

不要把 synthetic trace 的命中率或 tok/s 当作模型验收结果。

## 5. 结果目录建议

```text
m0-results/
  baseline-mmap.log
  baseline-none.log
  qwen35-route.jsonl
  layout.txt
  simulator-none.txt
  simulator-frequency.txt
  simulator-oracle.txt
```

建议保存完整命令行和机器信息：

```bash
uname -a > ./m0-results/system.txt
git rev-parse HEAD >> ./m0-results/system.txt
```

## 6. 常见问题

### 找不到 routed expert tensor

`matched_tensors=0` 通常表示模型不是 Qwen3.5 routed MoE、传入了不包含 tensor 的错误 shard，或 tensor 命名与当前 loader 约定不同。先用正确的模型文件和全部 split 重试。

### `expert ranges are not aligned`

这是 `--alignment N` 的检查失败。先不指定该选项确认普通 range layout 是否正确；不要据此直接开启 direct I/O。

### `row data validation failed`

检查模型文件是否完整、split 顺序是否正确，以及传入的 shard 是否与原模型属于同一组。不要忽略该错误进入后续 resident-slot 实现。

### trace 文件已存在

默认不会覆盖已有 trace。确认旧结果不再需要后加 `--force`。

### 当前没有 `--moe-streaming`

这是预期行为。v1.1 的 M1/M2 runtime 尚未进入当前提交；当前提交只验证 PR-1/M0 和 PR-2 的前置条件。

## 7. 下一步

当前 PR-3 仍是独立 blocking prototype。下一步应在不改变 exact 输出的前提下，把 range reader、generation/event 和 cache admission 拆成可测试的 M2 runtime，再接入默认模型 graph；在此之前不启用隐式 eviction、MTP 或 approximate skip。
