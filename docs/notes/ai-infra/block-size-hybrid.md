---
title: block_size 对混合注意力推理的影响
description: 模拟/量化 vLLM HMA 下 block_size 对 kernel、KV 传输、前缀缓存复用与显存尾块的影响
---

# block_size 对混合注意力推理的影响

> 模拟 / 量化：4 个实验台、18 个可复跑脚本。全部数字来自本地实测（RTX 4060 Laptop 8 GB, sm_89）与真实 trace 仿真。

## 问题背景

vLLM 的 HMA（Hybrid Memory Allocator）为了让异构缓存对象**共用一个物理池**，用最小公倍数把页大小统一 —— 代价是**全注意力层的 `block_size` 被推导出来、用户不能手动调节**。

分页注意力的分配单元通常 **16–32 token**，而混合注意力模型实测落在 **528 / 784 / 1568 / 2096**，差 **16 到 131 倍**。这不是「块稍大」，而是**把分配单元推到了传输单位才该有的量级**。

## (1) 结论

| 维度 | 最优 block_size | 偏离代价（实测） |
|---|---|---|
| 全注意力 kernel | **因配置而异**（8 个配置 8 种最优） | 32–512 内仅 3/8 配置在 1.10× 内；最差 **6.32×**（bs≤16）、**7.55×**（b1-4k@512） |
| KV 传输 | **越大越好** | 128k/RDMA400G：**803.6→410.5 ms（−48.9%）**；NVLink4：**927.2→56.2 ms（16.5×）** |
| 前缀缓存复用 | **与负载共享粒度对齐** | 同为 bs=528，损失从 **0.93%** 到 **100%** 不等 |
| 显存尾块 | **短请求偏好小块** | 8 GB 池、600-token 请求：并发 **95→55（−42%）**；bs=2048 尾块浪费 **28.7%** |

## (2) 实验与图

### ① kernel 模拟

自写 Triton paged-decode kernel（vLLM v2 结构），扫 16 种块大小 × 8 个配置。

![kernel：块大小 vs 解码延迟](/wiki/images/block-size/kernel_block_size_latency.png)

![kernel：lane padding](/wiki/images/block-size/kernel_lane_padding.png)

- **8 个配置出现 8 种最优块大小**，不存在通用最优；
- 32–512 内仅 **3/8** 配置互相在 1.10× 内；bs≤16 最差 **6.32×**；
- lane padding 占 **7.3%**（1.2155–1.3046 ms）；
- 后端白名单：FlashInfer 只接受 16/32/64，**528 与 1568 各丢 3 条路径**。

![kernel：后端白名单](/wiki/images/block-size/kernel_backend_whitelist.png)

### ② KV 传输模拟

解析模型（描述符 + 链路时延），并用真机 gather 基准标定（**100.3–103.7 GB/s**）。

![传输：块大小 vs 端到端时延](/wiki/images/block-size/kv_transfer_latency_vs_blocksize.png)

![传输：冲突区](/wiki/images/block-size/kv_transfer_conflict_zone.png)

- 128k 上下文、RDMA 400G：bs 16→1024 为 **803.6→410.5 ms（−48.9%）**；
- NVLink4：**927.2→56.2 ms（16.5×）**；
- 传输维度与 kernel 同向，**偏好大块**。

![传输：gather 带宽](/wiki/images/block-size/kv_gather_bandwidth_vs_blocksize.png)

### ③ 前缀缓存复用仿真

4 组真实 trace（Mooncake Conversation / Tool-Agent、LMCache Agentic、合成 Agentic），1200 请求 / 1685 万 token。

![复用：命中率 vs 块大小](/wiki/images/block-size/prefix_hit_vs_block_size.png)

![复用：512 vs 528](/wiki/images/block-size/prefix_512_vs_528.png)

| 负载 | 特征 | bs≤512 | bs=528 | 相对损失 |
|---|---|---:|---:|---:|
| Mooncake Conversation | 相邻请求共享**恰好 1 个 512-chunk** | 3.507% | **0.000%** | **100%**（590,848 token 全丢） |
| Mooncake Tool-Agent | 共享 1–N chunk，粒度混杂 | 27.574% | 23.434% | 15.3% |
| LMCache Agentic（10 session） | 相邻 iteration 共享 >99% | 89.342% | 88.512% | 0.93% |
| 合成 Agentic（200 session） | 同上但更短 | 88.243% | 86.171% | 2.35% |

- Conversation 型：**12,030 对相邻 `hash_ids` 全部只共享 1 个 512-chunk**，528 不整除 512 → 整段吃不到；
- 8/16/40 GB 三档预算下损失均为 **100%**；**51 个请求即需重算 25,088 token**。

### ④ 容量与尾块

分配器模型 + 8 GB 池仿真。

![容量：端到端并发](/wiki/images/block-size/e2e_capacity.png)

![容量：零填充分配器收益](/wiki/images/block-size/zeropad_allocator_gain.png)

- 块越大尾块浪费越多：bs=2048 时尾块浪费占真实 token 的 **28.7%**（bs=512 为 1.77%、34.9 MiB/请求）；
- 600-token 短请求并发随块增大 **95→55（−42%）**。

### ⑤ 四维合成（统一换算成 ms/请求）

32k 上下文 + 256 token 输出 + PD 分离 + RDMA400G。

![合成：四维权衡](/wiki/images/block-size/synthesis_tradeoff.png)

- 最优点 **bs=256**；
- 偏离代价：bs=16 **+443.8 ms**、528 **+31.6 ms**、1024 **+38.5 ms**、512 **+14.8 ms**、64 **+11.2 ms**。

## (3) 规则调研

**vLLM 混合注意力（HMA）**：统一页取各缓存对象 `block_size` 的 lcm，全注意力层的 `block_size` 因此被**推导**出来 —— 设计文档 Case 4 明写 "increase `block_size` until ≥ `state_size_mamba`"，并承认可能**超过 400、"too large"**；实测线上取值 **528**（Qwen3.5 hybrid）、**1568**、**2096**（Qwen3.5-397B/TP2）。

![可行性：lcm 网格](/wiki/images/block-size/lcm_feasibility_grid.png)

**同类工作均把粒度固定或推导**：AVMP 按张量形状推导、vTensor 恒 2 MB、GraniKV 默认取 1、DualPools 固定 512 —— **四家都没有把 `block_size` 当变量扫过**。

## 一句话

损失主要来自「**块大小与负载共享粒度不对齐**」，而不是块大小本身；**kernel/传输偏好大块，复用/尾块偏好小块**，四维合一的均衡点在 **128–256**。

## 适用范围（诚实边界）

- kernel 数字来自 8 GB 笔记本 GPU 上的自写 Triton kernel：**趋势可信、绝对值不可外推**；
- KV 传输是解析模型（本机无 RDMA/NVLink），用真机 gather 带宽作锚点；
- 528 的伤害是**负载相关**的，不能讲成通用灾难；
- 粒度 × 布局的正交实验**尚未运行**（需 1 张 ≥24 GB 卡），因此「粒度是一阶变量」止于四维曲线，暂不主张因果分离。
