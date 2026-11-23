# AI Infra

模型之外的所有基础设施：怎么把训练和推理跑满 GPU、怎么部署、怎么在有限显存下把吞吐和延迟做到能接受。

## 已有笔记

- [长上下文注意力：存与读的两本账](/notes/ai-infra/long-context-attention)：DSA / QSA / MSA / CSA / HCA 五种机制的独立图解讲义，含交互实验与预算计算器
- [block_size 对混合注意力推理的影响](/notes/ai-infra/block-size-hybrid)：vLLM HMA 下块大小由状态页推导且用户不可直接下调的代价，包含 kernel、KV 传输、前缀复用与显存尾块四个维度的实测与 12 张实验图
- [块大小能否修改、改了有什么影响](/notes/ai-infra/block-size-modifiable)：从源码、上游 RFC/issue、论文与其他引擎四条线索核对：`--block-size` 只能往大改，匹配粒度可用 `--prefix-match-unit` 单独下探，以及各自对显存与命中的影响
- [block size 实测进展](/notes/ai-infra/block-size-weekly)：Qwen3.5-2B/9B 上块大小推导、显存容量曲线（块翻倍容量降 32.7%）、吞吐性能、块内复用与 TP 对照实测
- [SWA/Full 模型 block size 实测进展（Gemma 4）](/notes/ai-infra/block-size-gemma4)：纯注意力（25 滑窗 + 5 全局）模型上块大小的端到端与注意力 kernel 实测（TTFT +16.6%、TPOT 持平、容量 −34.3%、prefill kernel +600%）
- [Qwen3.8 KV 预算与吞吐实测](/notes/ai-infra/qwen38-kv-budget)：AgentX 长上下文负载下每卡 KV 预算 1.0B0–1.3B0 的跨并发实测（全量输出口径：并发 16 未饱和 +13.3%、并发 20 超线性收益 +37.0% 达 559.7 tok/s、并发 24 过载跌至 144.8 tok/s）

## 计划中的笔记

- [ ] 推理引擎横评：vLLM / SGLang / TensorRT-LLM 在同一张卡上的实际吞吐
- [ ] PagedAttention 到底省了什么：KV cache 碎片与显存占用实测
- [ ] 连续批处理（continuous batching）的调度行为与尾延迟
- [ ] 单机多卡到 K8s 集群的部署路径：从裸机 script 到 operator
- [ ] 量化上线前的验收清单：精度回归、显存占用与吞吐测试
- [ ] GPU 利用率排查：nvidia-smi 显示 100% 但吞吐上不去的常见原因

## 待办

- [ ] 建一份「显存估算速查表」（参数 / 优化器状态 / KV cache / 激活值）
