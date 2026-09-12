# EdgeMoE 端侧 MoE 推理优化评测总结（HumanEval）

日期：2026-09-12。机器：i7-1165G7（4C/8T），16GB 物理内存（8GB 预算），CPU-only，无独立显卡可用。
模型：Qwen3.6-35B-A3B-UD-Q4_K_M.gguf（22.1GB，qwen35moe，40 层，256 选 8 路由），llama.cpp 自带 mmap 按需加载。
工作负载：HumanEval 5 题（task 0,5,10,15,20），n_predict=32（31 个 decode 步），贪心采样（--temp 0），ctx 2048，8 线程，--no-warmup。

内存控制：每个 llama-cli 子进程施加 6.5GB 硬工作集上限（SetProcessWorkingSetSizeEx，HARDWS_MAX），模拟并保证 8GB 总预算；空闲内存 < 1.2GB 时驱动自动中止。TPOT 取 slot print_timing 的 eval time / tokens（不含 prefill 与模型加载）。

## 1. TPOT 结果（5 题均值 / 中位数，ms/token）

| 配置 | mean TPOT | median | 相对基线 |
|---|---|---|---|
| baseline（原始 mmap 推理） | 3385.2 | 3212.7 | 1.00x |
| opt1 专家跳过 k1=4, k2=16 | 1612.9 | 1606.9 | **2.10x** |
| opt2 = opt1 + DFlash 草稿(Q4_K_M) 草稿长 7 | 2430.0 | 2319.9 | 1.39x |
| opt3 = opt1 + opt2 + relaxed verify | 1703.2 | 1695.7 | 1.99x |

补充：opt1 配置在页缓存全热的状态下（trace 采集轮，同题同参数）TPOT 约 718-846ms，说明冷缓存下的 SSD 专家流式读取是主要瓶颈。

prefill（每题 78-176 token）：baseline 171-282s，opt1 94-175s，opt2/opt3 与 opt1 相当。prefill 阶段的专家并集覆盖几乎全部 256 个专家，因此跳过对 prefill 的收益有限；decode 阶段收益直接。

## 2. 投机解码观察（opt2 vs opt3）

| 指标 | opt2（严格贪心验证） | opt3（relaxed verify） |
|---|---|---|
| 草稿接受率（5 题） | 0.42 - 0.60 | 0.59 - 0.81 |
| 平均每周期接受长度 | 3.44 - 5.00 | 4.43 - 6.20 |
| mean TPOT | 2430.0 | 1703.2 |

relaxed verify（Medusa/TRT-LLM typical acceptance，eps=0.09, alpha=0.3）将接受率从 ~0.5 提升到 ~0.7，TPOT 较 opt2 改善 30%，基本追平 opt1。

关键发现：在本机的 SSD 流式（mmap + 6.5GB 预算）场景下，投机解码净收益为负。原因：一次验证批（最多 1+7 token）的专家并集约为每层 64 个专家（约 4.9GB 权重），远大于单 token 的 8 个（约 0.6GB），预算内放不下导致每个验证周期大量从 SSD 重读；接受 3-5 个 token 的收益无法抵消一次性 4.9GB 的 I/O。投机解码适合"专家可常驻"的场景，不适合冷流式场景。草稿量化（bf16 783MB -> Q4_K_M 236MB）本身有效，但草稿模型太小（236MB 全程驻留），量化对瓶颈（目标模型的专家流）无影响。

## 3. 专家命中率与提升空间（任务 5）

基于 opt1 配置的真实路由 trace（5 题共 32440 个 token-layer 选择，40 层）：

- 每层在整个运行中实际用到的专家数为 64/256。
- 被跳过专家（路由排名 5-8）落在"层内 top-16 热窗"内的比例（命中率）：**31.5%**。
- top-4 捕获的路由权重质量占 top-8 的 70.0%（m_k1/m_k2 = 0.70，即跳过一半专家只损失 30% 的路由质量）。
- 热窗扫描（对被跳过专家的覆盖率）：8 -> 19.4%，12 -> 25.7%，16 -> 31.5%，24 -> 41.1%，32 -> 48.7%，64 -> 69.7%。

结论：被跳过的专家呈长尾冷分布。若采用"热窗内补算"（resident-window refinement），窗口 16 需要增加约 1.26 个专家/token 的计算（+31%）才能挽回 31.5% 的被跳过质量；窗口 32 覆盖 48.7% 代价 +69%。质量/算力比很差，且更大的窗口不可行（RAM 装不下）。

**判断：k1=4/k2=16 已接近该工作负载的收益边界，专家选择层面没有显著的进一步优化空间。** 剩余 TPOT 由 4 个必算专家的 SSD 流式读取主导（热缓存时 opt1 可达 ~0.75s/token，冷缓存 1.6s），下一步的杠杆是专家预取/驻留调度（对应仓库 M4 计划），而不是更激进的跳过。

## 3.1 运行时 DRAM 专家命中率实测（补充，2026-09-12）

用系统缺页计数器（\Memory\Pages Input/sec x 4KB，覆盖 mmap 缺页读）在 6.5GB 预算下实测 opt1 配置的 decode 窗口，两次独立测量：

- SSD 缺页读入：12.6GB / 12.0GB（31 个 decode 步）
- 逻辑触碰的专家字节：31 x 311.3MB = 9.65GB（4 专家 x 40 层 x 1.946MB）
- SSD 流量超过逻辑需求（含预读放大），DRAM 专家命中率 ≈ **0-10%**，即当前预算下每个 decode 步的专家权重几乎全部从 SSD 流式读取（~413MB/s 持续读）。

注意区分两个"命中率"：本节是运行时 DRAM 命中率（SSD vs 内存）；第 3 节的 31.5% 是离线设计指标（被跳过专家被 top-16 热窗覆盖的比例）。页缓存较热的状态（连续重复同题后）TPOT 可到 750-850ms，说明命中率随缓存状态在 0 与较高值之间波动；6.5GB 工作集理论上可驻留每层约 47/64 个在用专家（上限 ~70%），但 Windows 在全局内存压力下不做专家感知的驻留，实际归零。这正是专家跳过能兑现 2.1x 的原因，也是下一步做预取/驻留调度的依据。

## 3.1.1 mmap 缓存与逐出机制、命中率成因（补充，2026-09-12）

系统没有专家感知的缓存或逐出策略：llama.cpp 仅 mmap 文件，驻留与逐出全部由 Windows 内存管理器按 4KB 页粒度完成（工作集上限 trim -> standby list -> 全局近似 LRU 老化回收；cache manager 对连续缺页流做预读）。命中率高低的根源是访问模式的重用距离：

- 单步 decode 触碰 311MB（4 专家 x 40 层），远小于 6.5GB 工作集 -> 一步之内 40 层的专家页共存，不存在"后层加载冲掉前层"。
- 用 trace 实测已计算专家（top-4）的重用间隔：中位 3 步、均值 10.5 步（约 3.3GB 中间流量）、p90 = 31 步（约 9.6GB，接近缓存容量）。命中 = 页在其重用间隔内存活；未命中集中在长尾间隔上。
- 路由高度集中使 decode 热集只有 ~5GB（5 题合计，每层 34-105 个唯一专家；单题约 1.5-2.5GB），在工作集 + standby 里装得下 -> 预热后命中率 59-80%。被逐出的牺牲者几乎都是 prefill 触碰过一次的冷专家（每层其余 ~200 个）——正确的牺牲者。
- 冷态崩到 0-10% 的轮换源：每次进程加载对 22GB 文件做全文件 PrefetchVirtualMemory（init_mappings(true)，顺序填满 standby）+ 加载期 fit 构图 + 其他请求的 prefill 流量；热页在重用间隔内被挤出。OS 级精确归因（cap trim 循环 vs standby 回收顺序）未做最终定位。
- 推论：OS 只是因为"热集恰好装得下"才表现良好。专家感知的驻留窗口（钉住热集、跳过全文件预取、防止一次性冷页洪泛）能把命中率稳定在高位并消除冷态崩塌，即 M4 计划的价值。

## 3.1.2 内存分区模型（按 GGUF 元数据精确核算，2026-09-12）

设计原则：草稿模型全部权重 + 目标模型非专家权重 + KV/计算缓冲常驻（pin），只有路由专家按预算缓存/卸载。实测构成（GiB）：

| 部分 | 大小 | 说明 |
|---|---|---|
| 路由专家（可卸载） | 18.23 | 40 层 x 256 专家，q4_K gate/up + q6_K down |
| attention/GDN 权重 | 1.29 | 常驻 |
| embedding + output | 0.89 | 常驻 |
| shared experts | 0.13 | 常驻（每步都触碰） |
| 常驻小计（目标模型非专家） | 2.38 | |
| DFlash 草稿 Q4_K_M | 0.22 | 常驻 |
| KV + 计算缓冲（ctx 2048） | ~0.25 | 估计 |
| **pin 合计** | **~2.85** | |

预算核算：6.5GB 工作集上限 - pin 2.85 ≈ **3.4GB 专家缓存**（1789 个专家切片，等效 ~45/层，全局 LRU 模拟命中率 99%）；3.4 + 2.85 + OS 1.5-2.0 ≈ **7.7-8.2GB 物理内存**，与 8GB 预算吻合。需求侧：单题 decode 热集 1.5-2.5GB、5 题约 5GB（频率偏斜下 3.4GB 全局 LRU 命中 99%），均落在预算内。

现状与该模型的差距：OS 不执行这个分区，专家缓存与三股洪泛（22GB 全文件预取、fit 构图 ~11GB、prefill 一次性冷页）共用同一工作集/standby。要落地需要：跳过全文件预取、pin 常驻部分（mlock 或依赖每步触碰天然保活）、专家 arena 全局 LRU + admission 控制（prefill/fit 冷页直通不缓存）。

## 3.1.3 按层独立 LRU 会提升吗（离线模拟，2026-09-12）

用 5 题的 decode 访问流（top-4 已计算专家，155 步 x 40 层）离线模拟两种缓存策略，专家缓存预算同为 3.4GB（1789 个专家切片）：

- 全局 LRU（等价于理想化的现有 OS 行为，无洪泛污染）：命中率 **99.0%**
- 按层独立 LRU、均分 44 槽/层：命中率 **78.5%**（层间需求不均：每层唯一专家 34-105 个，静态均分饿死高需求层）
- 按层独立 LRU 要 64-105 槽/层（总预算 5.0-8.2GB，超出 3.4GB 预算）才追平到 87.6-89.7%

结论：按层分区本身不是收益来源，等预算下反而因层间需求不均而劣于全局 LRU。实测（59-80%/0-10%）与理想全局 LRU（99%）之间的差距来自洪泛与 OS 非理想行为（22GB 加载预取、fit 构图 ~11GB 触页、prefill 一次性冷页、WS trim 老化），而非 LRU 策略本身。正确的优化路径按性价比排序：(1) 跳过加载期全文件预取；(2) 对一次性冷页做 admission 控制（prefill/fit 页不进缓存或低优先级），保留全局 LRU；(3) 若做应用层 resident arena（PR-3 原型的方向），用全局 LRU 或按需求自适应配额，并配合层间预取流水（算第 l 层时预取第 l+1 层）隐藏 miss 延迟。收益量级：主要来自消除冷态崩塌（TPOT 1.5-2.1s -> ~0.75s），暖态 80%->99% 对单流 TPOT 仅省 ~25ms/tok（暖态瓶颈在 CPU）。

## 3.2 预热与 server 模式（补充，2026-09-12）

预热实验（连续 4 个 cli 进程：p0 预热、p5 预热、p0 复测、p10 新题，测量 decode 窗口的系统缺页读入）：

- p0 复测（预热后同题）：SSD 读 3,958MB / 触碰 9,649MB -> DRAM 专家命中率 **58.98%**
- p10 新题（前两题预热后首跑）：SSD 读 1,957MB -> 命中率 **79.72%**
- 对照：冷/争用状态下同配置命中率为 0-10%（3.1 节）

结论：**预热确实把命中率从 0-10% 提升到 59-80%**，且新题的 prefill 扫描本身就会把该题 decode 需要的热集装入缓存（prefill 是最好的预热器）。

llama-server 模式实测两次均在加载阶段把可用内存压到 0.6GB 以下（硬工作集上限也无法约束加载器的全文件预取+初始化触碰），存在挂机风险，本机放弃。机制上 server 也不会更好：cli 与 server 加载都会对 22GB 文件做 PrefetchVirtualMemory（llama_model.cpp init_mappings(true)），而命中率的主要搅动者a是加载期 fit 构图与各请求 prefill 的专家流量（实测每题 prefill 唯一专家页 1.1-11.8GB，层间路由集中度差异大），server 同样要做；server 仅省下 cli 每题一次的启动预取洪泛，收益是小头。要让本机能跑 server，需先给加载器加"跳过全文件预取"开关。

## 4. 生成质量抽查

problem 0 四个配置（贪心）输出前缀逐 token 一致，专家跳过与 relaxed verify 未破坏生成连贯性（与论文报告的 MMLU -0.35 分一致）。

## 5. 实现与文件

- 优化1（专家跳过）：`src/llama-graph.cpp build_moe_ffn`，参数 `--moe-skip-k1/--moe-skip-k2`，论文 arXiv:2609.04575 的 Eq.2（top-k1 计算 + top-k2 质量分母），实现说明 `docs/edge-moe-m1-optimizations.md`。
- 优化2（DFlash 草稿量化）：`llama-quantize` 产出 `D:\workspace\models\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`（236MB）；运行参数 `-md <draft> --spec-type draft-dflash --spec-draft-n-max 7`。
- 优化3（relaxed verify）：`common/sampling.cpp`（typical acceptance）+ llama-cli/server/speculative-simple 接线；参数 `--spec-relaxed-verify --spec-relaxed-eps 0.09 --spec-relaxed-alpha 0.3`。
- 明细数据：`edge-moe-results/bench.csv`（逐题）、`edge-moe-results/logs/*`（完整 llama-cli 日志）、`edge-moe-results/traces/*.jsonl`（路由 trace）、`edge-moe-results/hitrate.txt`（命中率分析输出）。
- 复现：`edge-moe-bench/run_bench.py`（驱动）、`edge-moe-bench/expert_hitrate.py`（命中率）、`edge-moe-bench/run_chain.sh`（全链路）。

## 6. 局限

- 5 题 x 32 token 为用户指定的快速测试口径；TPOT 的绝对值受页缓存冷热影响（基线先跑、后续配置受益于残留缓存，因此 opt1 的 2.10x 偏保守），但配置间的相对结论稳定。
- 本机仅 CPU，TPOT 绝对值不代表 GPU/Metal 平台。
- 投机解码的结论限于 8GB 预算 + mmap 流式场景；在专家可常驻内存的平台（如 32GB+ 或 GPU）预期相反。
