# EdgeMoE M1 端侧推理优化实现说明

本文记录在 llama.cpp 之上实现的三项端侧 MoE 推理优化、其参数语义与评测方法。实现均通过 CLI 参数开关控制，默认关闭时不改变 llama.cpp 原有行为。

## 1. 优化1：专家跳过（expert skipping）

论文：arXiv:2609.04575（Qwen3.6-35B-A3B，256 选 8 路由）。

方法：每层路由器仍输出完整 softmax 与 top-k 选择（本模型 k=8），但只精确计算 top-k1 个专家的 FFN；被跳过专家的概率质量保留在权重分母中：

```text
标准重归一化:  w_i = p_i / sum_{j in top-k} p_j          (权重和为 1)
本文方法:      w_i = p_i / sum_{j in top-k2} p_j,  i in top-k1
```

k2 > k1 时权重和 = m_k1/m_k2 < 1，是一个逐 token、输入自适应的降缩放，抵消了低 k1 下标准重归一化带来的增益膨胀（论文：(k1,k2)=(4,16) 在 35B 模型上 MMLU 仅损失 0.35 分，而 k2=k1 的标准重归一化损失 4.65 分）。

实现：`src/llama-graph.cpp` 的 `build_moe_ffn`。保留完整 top-8 选择用于路由 trace；对权重做 top-k1 切片，分母取 top-k2 的概率质量和（`ggml_argsort_top_k` + `ggml_get_rows` + `ggml_sum_rows`），并跳过 `norm_w` 重归一化。mul_mat_id 的形状从 8 降为 k1，路由专家 FFN 计算量与读入字节数同步减半。

参数：

- `--moe-skip-k1 N`：每层只计算 top-N 个路由专家（0 = 关闭）；
- `--moe-skip-k2 N`：权重分母取 top-N 概率质量（0 = 取 k1）。

实验配置：k1=4，k2=16。

## 2. 优化2：DFlash 草稿量化 + 草稿长度 7

DFlash（block-diffusion 草稿）由本 fork 原生支持：草稿 GGUF 架构为 `dflash`，`--spec-type draft-dflash` 会自动识别，每步从 `[id_last, mask x (block_size-1)]` 就地去噪生成最多 block_size-1 个草稿。

量化：`llama-quantize` 将 `Qwen3.6-35B-A3B-DFlash-bf16.gguf`（783 MB）转为 `Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`（236 MB），约为 bf16 的 30%。

参数：`-md <draft.gguf> --spec-type draft-dflash --spec-draft-n-max 7`（草稿长度 7）。

## 3. 优化3：投机解码 relaxed verify

参考 TensorRT-LLM / Medusa 的 typical acceptance：草稿 token 不再要求与目标模型贪心采样严格一致，而是在该位置的原始目标分布上满足

```text
p_target(draft_token) > min(eps, alpha * exp(-H))
```

其中 H 为该位置目标分布的熵（nats）。被拒绝的位置仍从目标模型采样纠正 token，与贪心验证的回退语义一致。语法约束（grammar）激活时自动退回严格贪心验证。

实现：`common/sampling.cpp` 的 `common_sampler_sample_and_accept_n` relaxed 重载；llama-cli（内嵌 server 路径）与 speculative-simple 均已接入。

参数：`--spec-relaxed-verify`（开关），`--spec-relaxed-eps P`（默认 0.09），`--spec-relaxed-alpha P`（默认 0.3）。

## 4. 评测方法

- 数据集：HumanEval（openai/human-eval），按用户要求取 5 题（task 0,5,10,15,20）。
- 每题一个 llama-cli 进程：`--no-warmup --perf --verbose --single-turn --temp 0`，n_predict=32（31 个 decode 步），ctx 2048，8 线程。
- 内存预算：驱动脚本对每个 llama-cli 子进程设置 6.5 GB 硬工作集上限（`SetProcessWorkingSetSizeEx`，QUOTA_LIMITS_HARDWS_MAX_ENABLE），模拟并保证 8 GB 总预算（进程 + 系统）。空闲内存低于 1.2 GB 时中止。
- TPOT：取 slot `print_timing` 的 `eval time / tokens`（不含 prefill 与加载）。
- 为什么用 llama-cli 而不是 llama-server：server 加载期的 memory fitting 会瞬时耗尽内存（曾打到 1.3 GB 空闲），逐题 llama-cli 进程边界使提交内存即时释放。

## 5. 结果

见 `edge-moe-results/bench.csv` 与 `edge-moe-results/summary.md`。
