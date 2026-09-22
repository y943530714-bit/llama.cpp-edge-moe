# EdgeMoE-Llama M0 / PR-3 与 expert arena 运行手册

本文对应 `edge_moe_design_v1.1.md` 的 M0–PR-3，并说明固定容量 expert arena 的运行方式。arena 默认关闭，只有显式传入 `--moe-arena-mib` 才改变 routed expert 的执行路径。

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
- 默认 CPU 推理路径中的固定容量 expert arena：路由完成后同步加载缺失的 `(layer, expert)` bundle，改写一份物理 slot ID，并按全局路由频次驱逐未被当前层引用的 slot；同频时优先驱逐最久未使用项。
- arena 在同一 context 的多次 decode/request 间保留；当前层用到的 slot 通过 refcount 固定到下一层 safe point。
- `llama_get_memory_breakdown()` 将 arena 的实际 buffer 大小计入 host context 内存。

尚未实现：

- 异步 I/O 和 P0/P1/P2 调度；
- Gate-First split graph、prefill double buffer 和 MTP prefetch；
- `.moepack` sidecar。
- LoRA、per-expert scale、非 CPU layer 和 `--parallel` 大于 1 时的 arena 路径。
- 对非专家权重做 OS 级物理内存锁定；普通模型权重仍由现有 mmap/backend 管理。

当前 arena 是应用层固定私有 buffer，不等同于 `mlock`/`VirtualLock`，操作系统仍可在内存压力下换出这些页。

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

### 4.8 在默认推理路径启用 expert arena

arena 的容量是硬上限参数，单位 MiB。它缓存完整的 `(layer, expert)` bundle，而不是只按 expert 编号跨层共享。当前实现要求 CPU-only routed expert layer 和单序列 context。启用后会自动关闭 CPU weight repacking，因为 arena 中保存的是 GGUF 原始量化布局：

```bash
./build/bin/llama-cli \
  --model /path/to/model.gguf \
  --parallel 1 \
  --moe-arena-mib 3400 \
  --ctx-size 8192 \
  --prompt "Explain expert caching." \
  --predict 128
```

启动日志会报告 requested budget、实际 allocated bytes、单 slot 大小和 slot 数。退出时会报告 hits、misses、evictions 和从原始 GGUF 映射区复制的字节数。实际分配量不会超过请求预算；预算不足以同时容纳一层全部 expert 时，context 初始化会明确失败。直接调用 C API 的程序也必须用 `use_extra_bufts=false` 加载模型；不兼容的 `CPU_REPACK` tensor 会被明确拒绝。

路由 tensor 仍保留逻辑 expert ID，权重、bias 和概率归一化继续使用逻辑 ID。只有供 expert matrix multiplication 使用的副本会在调度回调的 safe point 被换写成物理 slot ID。因此不需要拆成两张计算图。

当前同步 loader 从模型的 host-accessible mmap 地址复制 slice。它实现了真实、固定容量、可驱逐的应用层 cache，但没有绕过 OS page cache，也没有异步预取。做严格 8 GiB 物理内存验收时，应在 8 GiB VM 或 Job Object 等硬限制环境中另外测量 committed/private working set 和 page fault；不能只看宿主机的 RSS。

### 4.9 验证 Windows unbuffered I/O 和非专家常驻

第一阶段的 probe 可以用独立的 Windows unbuffered handle 读取专家 slice，并与普通文件读取结果逐字节比较。多个 `--experts` 会作为一批 overlapped 请求提交：

```bash
./build/bin/llama-edge-moe-slot-probe \
  --model /path/to/model.gguf \
  --layer 0 \
  --part gate_up \
  --experts 0,1,255 \
  --slots 3 \
  --direct-io
```

成功时最后一行包含 `direct_io_summary` 和 `compare=ok`。`requested_bytes` 是实际专家数据量，`transferred_bytes` 包含扇区对齐后多读取的字节。

以下模式加载完整模型但关闭预取，然后在给定的进程 Working Set 上限内只锁定非专家 tensor。该参数仅属于 probe，不能用于正式推理：

```bash
./build/bin/llama-edge-moe-slot-probe \
  --model /path/to/model.gguf \
  --resident-budget-mib 8192
```

成功时 `residency_summary` 会报告持久化的专家源描述数量、锁定的非专家字节和锁定前后的进程 Working Set。失败时不会改用普通缓存 I/O，也不会把专家 tensor 一起锁定。

### 4.10 运行预算受控的 decode streaming cache

Windows CPU-only、单并发场景可以用总进程 Working Set 目标启用正式 runtime。当前本机 10 GiB 推荐配置如下：

```powershell
.\build-msvc-low\bin\Release\llama-cli.exe `
  -m D:\path\to\target.gguf `
  -md D:\path\to\draft.gguf `
  --spec-type draft-dflash --spec-draft-n-max 3 `
  --moe-streaming-budget-mib 10240 `
  --moe-streaming-io-depth 32 `
  --moe-streaming-layered-cache `
  --moe-skip-k1 4 --moe-skip-k2 16 `
  --parallel 1 -ngl 0 -c 2048 -t 8 -tb 8 --no-warmup
```

该参数会强制 target 使用 mmap、关闭通用预取和 weight repack，并关闭自动 memory fitting。runtime 在给定预算内先锁定非专家权重，再扣除 DFlash 文件大小和 768 MiB 运行时保留量，剩余空间按实际 expert bundle 大小转换成 slot 数。专家 miss 只经过 Windows unbuffered overlapped I/O，不从专家 mmap 页复制，也不会退回 buffered I/O。8 GiB 服务只需把预算改为 `8192`，但必须接受更少的专家槽和更高的 SSD miss。

退出时会报告 hot/cold cache 的 hit、miss、eviction、promotion、demotion，unbuffered I/O 的请求数、逻辑/物理读取量和等待时间，以及进程当前/峰值 Working Set。`--moe-streaming-io-depth` 控制同时在途的 Windows unbuffered 读取，默认值为 8，可在 1-64 之间调节；本机 NVMe 实测推荐 32。`--moe-streaming-budget-mib` 与旧的 `--moe-arena-mib` 互斥。

streaming runtime 会区分 prefill 与 decode/verify。每个新请求的第一次 prefill 会取消上一个请求的 resident 保护，然后按本次 prefill 的逐层 router probability 总和选择 resident，未入选专家仅在本层计算期间占用临时 slot。最终路由权重在原有 arena ID safe point 一并读取，不增加一次计算图同步。decode/verify 的一次性冷专家只能复用动态槽；同一请求内第二次命中的专家才有资格晋升热槽。热槽淘汰按本请求累计路由权重乘以每个 decode batch 0.70 的时间衰减选择最低分，同分再按 LRU。退出日志分别报告 prefill 和 decode/verify 的命中率、读取量与 I/O 等待时间。

`--moe-streaming-layered-cache` 启用分层固定槽位。10 GiB 配置实测分配 3175 个 slot，40 层各有 79 或 80 个 slot；自动冷热划分为每层 60 个热槽和 19 或 20 个动态槽。prefill 先装入本层按累计 router score 排名的热专家，再装入非热点临时专家；decode 的装入、晋升和淘汰均限制在当前层，避免不同层争抢全局 resident。该模式要求同时设置 `--moe-streaming-budget-mib`。

`--moe-streaming-hot-slots-per-layer N` 可以覆盖自动冷热比例，取值 0 保留自动划分；显式配额至少为每层留下一个动态槽。8 GiB 的四条 GSM8K 与四条 HumanEval trace 仿真及实测显示最优比例与请求有关：38/15 在 GSM8K 仿真最优，但实测比自动 40/13 慢 0.84%；32/21 在 HumanEval 实测将总时延降低 2.28%，却使 GSM8K miss 增加 152 次。因此通用配置仍使用自动划分，显式参数仅用于固定工作负载的离线调优。

`--moe-streaming-hot-slots-by-layer N0,N1,...` 可以进一步给每层设置精确热配额，列表长度必须等于模型层数，并与统一配额参数互斥。该能力用于离线 trace 调优，不是通用推荐配置。当前 40 层实验表在八条 held-out GSM8K 上只减少 2 次 decode miss，SSD wait 反而增加 1.00%；在八条 held-out HumanEval 上减少 244 次 decode miss，总时延降低 1.60%。跨层共享 overflow 的实测更差，已从 runtime 撤回。因此默认仍使用自动配额；逐层表只适合明确偏向代码负载且完成独立验证的服务。

另有两个默认关闭的实验开关：`--moe-streaming-prefill-full-layer` 复用动态槽组成双缓冲，在 512 token 以上的主 prefill 执行 full-layer K+1；`--moe-streaming-decode-prefetch` 用本请求 prefill 学到的相邻层专家共现关系，在 K 层计算期间预取 K+1 层一个候选。小样本实测二者均未超过分层固定槽：full-layer 在 304-token 样本上使 prefill 增加 18.73%，decode 预取使 GSM8K/HumanEval 总时延分别增加 3.43%/0.76%。因此当前推荐命令不添加这两个开关。

阶段识别按标准 server batch 语义进行：prefill 的输入 token 数大于输出数，speculative verify 为每个候选请求输出。自定义调用方若为 prompt 强制打开 logits-all，应先补充显式阶段标记，不能依赖这一自动判定。

8 GiB 物理内存压力复现可使用 `edge-moe-bench/hold_available_memory.py --available-mib 8192`。该进程会触碰并 `VirtualLock` 占位页，打印 `READY` 后保持到 stdin 关闭；如果不能把可用内存压到目标容差内会直接失败，不会静默退化成可换出的普通分配。`profile_spec_phases.py --mode mmap|streaming` 会采集每个请求的 draft、target verify、Working Set、硬页读，以及 streaming 的逐 batch cache/I/O 增量。

`profile_spec_phases.py --spec-draft-n-max N` 可以扫描 DFlash 投机长度。当前 10 GiB strict verify 小样本中，GSM8K 的 `N=3/5/7` decode 速度分别为 14.27/13.16/11.79 tok/s，HumanEval 分别为 15.03/15.07/14.86 tok/s；HumanEval 的 `N=5` 复测达到 15.39 tok/s，但端到端差异很小。综合两类负载当前推荐 `N=3`。推荐配置不启用 relaxed verify，避免因验收规则变化使不同投机长度生成不同内容。

### 4.11 运行离线 cache simulator

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

### `--moe-streaming-budget-mib` 启动失败

确认平台是 Windows、模型层全部在 CPU、`--parallel 1`，并且没有同时传 `--moe-arena-mib`。该模式要求 target 为 mmap 加载；显式的 `--load-mode none`、`mlock`、`mmap+mlock` 或 `dio` 会被拒绝。预算不足以容纳非专家常驻集、运行时保留量和至少一层专家时也会严格失败。

## 7. 当前状态与限制

第二版 router probability + recency 策略已完成。两次 8 GiB 物理内存压力复测得到完全相同的 cache 结果：相对第一版，decode 命中率从 78.322% 提升至 78.614%，少 46 次 SSD miss；两次时延均值的总 wall time 降低 2.85%，target verify 降低 2.70%，decode I/O wait 降低 2.18%。相对原 layer-batched LRU，总 wall time降低 13.86%。首次出现即按分数晋升的试验虽然把命中率推到 78.697%，但造成 prefill miss 从 3165 增至 4487，因此未保留。

10 GiB 相对 8 GiB 的 GSM8K 总时延从 9.077 s 降至 7.536 s，decode expert hit rate 从 74.67% 升至 84.86%，SSD wait 从 2.828 s 降至 1.796 s；HumanEval 总时延从 6.848 s 降至 6.277 s，hit rate 从 70.43% 升至 80.85%，SSD wait 从 1.609 s 降至 1.117 s。这些数字来自本机少量样本，用于选配置，不代表完整任务集的统计结论。

当前实现只支持 Windows、CPU target、单并发和 host-accessible GGUF。MTP、LoRA、并行序列和非 CPU backend 必须分别建立正确性与生命周期验证后才能接入。
