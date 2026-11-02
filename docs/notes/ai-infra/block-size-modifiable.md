---
title: 混合注意力模型的块大小：能否修改、改了有什么影响
description: 从 vLLM 源码、上游 RFC/issue、论文与其他引擎四条线，回答 lcm 推导出的块大小能否被修改，以及改动对显存、命中率与延迟的影响
---

# 混合注意力模型的块大小：能否修改、改了有什么影响

> 源码基线 vLLM `main` @ `b10a8c2c`。结论按"源码 → 上游讨论 → 论文 → 其他引擎"四条线交叉核对，每条附出处。

## 结论速览

| 问题 | 答案 |
|---|---|
| 块大小怎么来的 | 由 Mamba 状态页字节数反解，**不是**直接取最小公倍数；多组时才 lcm |
| 能否指定 | 能，但只能**往大**改；小于推导值会被抬回去 |
| 能否把匹配粒度改细 | 能，`--prefix-match-unit`，但必须整除块大小，且**默认不生效** |
| 528 是哪个参数的值 | 是 `block_size` 的值，不是命令行参数，而是 `--block-size` 被推导覆盖后的结果 |
| 只改块大小有实测吗 | **没有**。公开数字改的都是共享前缀长度、匹配粒度或分配结构，块大小始终固定 |
| 那它们说明了什么 | 块大小与负载共享前缀的对齐关系决定命中：块大小固定为 528 时，命中可在 54.0% 与 25.5% 之间摆动 |
| 有没有人系统扫过这一维 | 论文页大小全为常量；GitHub 上 `prefix-match-unit` 的 1,094 条命中全是单点配置 |

## 一、块大小到底怎么算出来的

旧说法"取各组块大小的最小公倍数"不准确。真正入口是 `vllm/platforms/interface.py` 的 `Platform._align_hybrid_block_size`：

```python
attn_block_size = alignment * cdiv(mamba_page_size, alignment * attn_page_size_1_token)
```

即用 Mamba 状态页字节数除以"每 token 注意力页字节数"，向上取整得到注意力块大小；然后强制 `mamba_block_size = block_size`，再把 Mamba 页 padding 到注意力页。**只有存在多个缓存组时**，才由 `resolve_kv_cache_block_sizes` 取 LCM（`vllm/v1/core/kv_cache_utils.py`）。

因为 `get_num_kv_heads(parallel_config)` 参与计算，所以**块大小随张量并行度变化**——这解释了为什么同一模型换 TP 值就跳。

上游 PR #64624 给出了更宽的表述：`lcm(kernel alignment, backend alignment, prefix_match_unit)`，即推导输入除了 Mamba 状态，还包含 kernel 与后端的对齐要求。

启动时可以直接从日志读实际值：

```
Setting attention block size to <N> tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by X%
```

### 528 对应的是哪个参数

同一个值可以在四个层次上看到，写实验记录时要指明用的是哪一层：

| 层次 | 名称 | 说明 |
|---|---|---|
| 命令行 | `--block-size` | 用户输入；混合模型下可能被忽略或被抬高 |
| 配置字段 | `CacheConfig.block_size` | 最终生效值；另有 `user_specified_block_size` 标记用户是否设过 |
| 运行期字段 | `scheduler_block_size` / `effective_attention_block_size` / `hash_block_size` | 调度器用的块大小、实际注意力块大小、哈希粒度 |
| 日志 | `Setting attention block size to 528 tokens ...` | 只在用户值被抬高时打印，用户值更大时静默 |

混合模型下还有一条镜像关系：`mamba_block_size = block_size`，即 528 同时也是 Mamba 侧的块大小。

## 二、已观测到的取值跨度比想象大

| 块大小 | 模型 / 条件 |
|---:|---|
| 400 | Qwen3.8-Flash-Next |
| 528 | Qwen3.5（4B 档） |
| 544 | Qwen3.5-0.8B |
| 768 | Kimi K3（TP8） |
| 784 | Qwen3.6-27B / Qwen3.8-27B |
| 896 | Qwen3.6-27B（某 CPU 平台） |
| 1600 | Qwen3.8-27B（int8 GPTQ） |
| 1888 | Kimi-Linear（TP1） |
| 2096 | Qwen3.5-397B-A17B-FP8（TP2） |
| 2112 | Qwen3.5-35B-A3B（fp8） |
| 12288 | Kimi K3（DCP8 + fp8，MLA 组） |

所以描述区间时不要写"528–2096"，那是 2026 年上半年的观测范围。

## 三、能改的四个旋钮

| 旋钮 | 作用 | 默认 | 约束与已知坑 |
|---|---|---|---|
| `--block-size` | 指定物理块大小 | 空（自动推导） | **只能往大**。源码注释 "may override user settings"，代码只在用户值小于推导值时抬高；调大后命中粒度变粗、Mamba 页 padding 变大 |
| `--prefix-match-unit` | 只改前缀匹配粒度，不动物理页 | 各组块大小的 GCD | 混合模型下 GCD 等于块大小，所以**不显式设置就完全无效**；必须整除块大小，否则启动报错 |
| `--mamba-block-size` | 指定 Mamba 块大小 | 空 | 只在开启前缀缓存时可设；它会改变推导出的统一块大小（对推导值的影响需实测确认） |
| `prefix_cache_retention_interval` | 每 N token 保留一个检查点 | `0`（只留可复用边界） | 保留越多命中越好、显存越多；约 (K−1)×2–8 MiB/请求 |
| `--mamba-cache-mode` | 旧开关 | — | **已是弃用空操作**，`all` 模式已被删除，不要当实验自变量 |
| `--attn-pack-size` | 多个全注意力层共页以缩小块大小 | — | **从未发布**，对应 PR 已关闭未合并 |

**`--prefix-match-unit` 的因子陷阱**：它必须整除块大小。按因数分解，528 = 16×33、784 = 16×49、944 = 16×59、2096 = 16×131，**这些值上只有 16 可用**；544 可用 16/32，768 可到 256，896 可到 128，1600 可到 64。文档里常举的 64 在 528/784/944/2096 上会直接启动报错：

```
Invalid prefix_match_unit=64; all KV cache block sizes must be divisible by prefix_match_unit
```

## 四、改了对显存的影响

显存影响分三层，不能混为一谈。

**第一层：页 padding。** 块变大时 Mamba 页要 padding 到注意力页，启动日志会打印 padding 百分比。这一层是确定的浪费，但比例通常不大。

**第二层：检查点保留。** align 模式下每个请求只保留少量 Mamba 状态检查点。要把命中粒度做细，就得保留更多检查点。上游估算每多保留一个检查点约 **2–8 MiB/请求**（Qwen3.5-4B 量级），默认 `0` 即不保留。

**第三层：容量核算。** 混合模型的 KV 显存曾被**过估约 7 倍**（7.57 GiB 对约 1 GiB，issue #37121），原因是按统一页核算时把 Mamba 状态摊到每个 token 上。反向的两个数据点：

| 改动 | 显存/容量变化 | 出处 |
|---|---|---|
| 填满 padded page | KV 容量 **+15.6~16.0%** | PR #63347 |
| 取消统一页、改为每组独立页 | 容量差 **<1%** | RFC #64401 |

也就是说：**统一页本身并不怎么浪费容量（<1%），真正的浪费在 padding 与检查点保留策略上。**

## 五、改了对命中率与延迟的影响

先分清每项实验实际改的是什么变量。**下面没有一行是"只改块大小"**。

| 实验 | 实际改的变量 | 块大小 | 结果 |
|---|---|---|---|
| 共享前缀 1600/600 → 1500/700 | **共享前缀长度** | 528，固定 | 命中 52/64 → **0/64**，TTFT 433 → **905 ms** |
| 同上，当前 main 复测 | 共享前缀长度 | 528，固定 | 命中 54.0% 对 25.5%，TTFT 245.95 对 300.25 ms |
| 加 `--prefix-match-unit 400` | **匹配粒度** | 1600，固定 | 40K 上下文 TTFT 5.56 s → 2.79 s |
| 投机解码 + `--prefix-match-unit 128` | 匹配粒度 | 固定 | 命中永久 0（反例） |
| 检查点保留策略 | **检查点密度** | 固定 | 命中 0% → 74.3%，TTFT p50 86.0 s → 4.5 s |
| PR #46251 分配解耦 | **分配结构** | 逻辑块不变 | 命中 +32%，prefill 2.7×（作者自测） |
| PR #63347 填满 padded page | **页 padding** | 逻辑块不变 | KV 容量 +15.6~16.0% |
| RFC #64401 取消统一页 | **分配结构** | 逻辑块不变 | 容量差 <1% |

因此这些数字能支撑的结论只有两条：一是块大小与负载共享前缀的对齐关系决定命中；二是改匹配粒度或分配结构可以缓解，但那都不是改块大小。

机制是几何的：align 模式下 Mamba 检查点只落在"提示末尾之前最后一个块边界"，如果这个位置落在各请求不同的后缀里，混合协调器要求所有组都命中，于是注意力组即使完美匹配也会被否决，命中整体归零。触发条件是 `floor((prompt_len-1)/block_size)*block_size > shared_prefix_len`。

## 六、上游与同行做到哪一步

**上游讨论（vLLM）**

| 编号 | 内容 | 状态 |
|---|---|---|
| RFC #45702 | 提出把哈希粒度与物理块解耦，支持块内部分命中 | RFC，拆成多个 PR |
| issue #45238 | 粗块大小导致命中归零，含上述实测 | Open，已有多方独立复现 |
| PR #46251 | 分配解耦：Mamba 申请连续内存，块回到自然值 | Draft，作者自测命中 +32%、prefill 2.7× |
| PR #45939 / #46384 / #49502 | `--prefix-match-unit` 的实现 | 已合并 |
| PR #35703 / #37429 | 全注意力层共页 / Mamba 独立池 | 关闭未合并 |

**论文**

- **Jenga（SOSP'25，arXiv 2503.18292，DOI 10.1145/3731569.3764823）是 vLLM 这套方案的原论文。** 它比较过 GCD/MAX/LCM 三种兼容页，指出 MAX 页的病理："Jamba 52B 要求每个自注意力页 1344 token，超过 ShareGPT 平均请求 1085 token"；但对 LCM 自身的命中损失只写了一句"未观察到性能退化"，**没有量化**。这是目前最明确的空白。
- **DualPools** 给出了一组端到端数字，但要注意它改的不是块大小：attention 侧的 512-token 块全程固定（论文原话 `held fixed … without per-organization optimization`），改的是 recurrent 状态的**页构成**——从 1557 KiB 的统一大页拆成 3 个 512 KiB 成员页加一个 conv 页。结果是吞吐提升 47.6%~56.2%，b63 处 TTFT p99 从 3.687 s 降到 0.578 s。它的另一个设计点（原生 16-token）在大预算下反输 12.6%，说明页结构与预算耦合，但这不能读成"块大小收益不单调"。
- **Marconi（MLSys'25）是不改粒度的对照组**：靠准入策略把命中提升 4.5~34.4×，同时是唯一量化过"块大小 → 复用率"的工作（block 32 时 KV 块复用 25.0%，而 SSM 状态复用仅 0.4%）。
- **AVMP（arXiv 2605.22416）**做两级分页：底层细页，上层合并成注意力块或 Mamba 块。
- **GraniKV 已撤稿**（作者声明吞吐测量方法有误），其 page-size-1 的收益数据不可引用。

**其他引擎**

| 引擎 | 页/块参数 | 关键事实 |
|---|---|---|
| SGLang | `--page-size`，默认 **1** | 官方 blog 把"混合模型支持 page_size > 1"列在 Future Work；源码已有按组不同粒度的抽象 `KvBufferDesc(row_bytes, tokens_per_row)` |
| TensorRT-LLM | `tokens_per_block` | 必须是 2 的幂且 >1；自己承认池间静态配额 "This is not optimal" |
| LMDeploy | `cache_block_seq_len` | 源码默认 64，文档写 128（文档滞后）；PyTorch 引擎要求 kernel block ≥16 且为 2 的幂 |
| Furiosa | 无页参数 | 全局注意力池 + 滑动窗口池自动分区 |

**没有任何引擎提供"每组独立页大小"的参数**，都是用"分组建池 + 组间配额"吸收异质性。vLLM 的统一页是它自己的取舍，而 RFC #64401 正在讨论移除。

## 六点五、真正的空白：没人只改块大小

把已有工作按"改了什么"归类，可以看出空位在哪：

| 研究线 | 改的变量 | 块大小 |
|---|---|---|
| 论文（9 篇） | 分配结构、页布局 | 全为常量 |
| vLLM issue | 共享前缀长度、匹配粒度 | 固定 |
| 上游 PR | 分配结构、页合并 | 逻辑块不变 |
| 社区配置 | 单点取值 | 无双值对照 |
| **本实验** | **只改块大小** | **自变量** |

GitHub 代码搜索 `"prefix-match-unit"` 命中约 1,094 条，全部是单点配置，没有两条不同块大小的对照。这就是可以补的位。

## 七、如果要自己测，三个自变量要拆开

1. **物理块大小**：`--block-size`，只能 ≥ 推导值；
2. **匹配粒度**：`--prefix-match-unit`，必须整除块大小；
3. **检查点保留**：`prefix_cache_retention_interval`，直接换显存。

三者混在一起测，结果无法归因。最小矩阵：

| 维度 | 取值 |
|---|---|
| 块大小 | 推导值、推导值×2、推导值×4 |
| 匹配粒度 | 默认（无效）、块大小的因子（如 528 → 264/176/132/16） |
| 保留策略 | 0（默认）、大于 0 |
| 负载 | 共享前缀长度分别取 块大小−1 / 块大小 / 块大小+1 |
| 指标 | 命中 token、TTFT、TPOT、KV 块数、`vllm:kv_cache_usage_perc` |

现成工具：

- `vllm bench sweep serve --serve-params <json>` 可以把 `{"block-size": 528}` 直接拼成命令行参数，自动重启服务、压测、落 JSON、按组合建目录（**但没有现成的块大小扫描配置，要自己写 JSON**）；
- 内置 `prefix_repetition` 数据集的共享前缀长度精确等于 `--prefix-repetition-prefix-len`，扫"块大小 ±1"能干净复现"差一 token 归零"；
- 正确性门槛用 LMCache 的 `hma_lm_eval`（gsm8k 两遍，中间 reset 前缀缓存，要求分数一致；GDN 混合模型不保证逐 token 一致）；
- 注意 `benchmarks/benchmark_serving.py` 在当前版本已是 deprecated shim，直接跑会 exit 1，要用 `vllm bench serve`。

## 八、四个容易踩的误读

1. **"块大小是 lcm 算的"**：只在多组时才 lcm，主路径是状态页反解。
2. **"用户完全不能指定"**：能往大改，只是不能往小改；匹配粒度还能独立下探。
3. **"672 这种奇怪的块大小对性能没影响"**（PyTorch 博客）：那是**吞吐**口径。块大小只是调度层的逻辑块，kernel 仍按 16/32/64 重新分页，所以"吞吐不变"与"命中变差"可以同时成立。
4. **"大页是行业惯例"**：SGLang 默认 page_size=1，page_size>1 还在计划里；没有引擎支持每组独立页大小。

## 参考

- vLLM 源码：`vllm/platforms/interface.py`（`_align_hybrid_block_size`）、`vllm/v1/core/kv_cache_utils.py`（`resolve_kv_cache_block_sizes`）、`vllm/config/cache.py`
- [vLLM RFC #45702：Partial Cache Hits for Hybrid Models](https://github.com/vllm-project/vllm/issues/45702)
- [vLLM issue #45238：粗块大小导致命中归零](https://github.com/vllm-project/vllm/issues/45238)
- [vLLM issue #37121：混合模型 KV 显存过估](https://github.com/vllm-project/vllm/issues/37121)
- [vLLM PR #46251：分配解耦（草案）](https://github.com/vllm-project/vllm/pull/46251)
- [LMCache：Hybrid Attention Models](https://docs.lmcache.ai/mp/hybrid_models.html)
- [PyTorch blog：Hybrid Models Meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- Jenga：arXiv 2503.18292（SOSP'25）
- AVMP：arXiv 2605.22416
- Marconi：arXiv 2411.19379（MLSys'25）
- vLLM 文档：[Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
