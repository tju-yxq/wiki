# AI Infra

模型之外的所有基础设施：怎么把训练和推理跑满 GPU、怎么部署、怎么在有限显存下把吞吐和延迟做到能接受。

## 已有笔记

- [长上下文注意力：存与读的两本账](/notes/ai-infra/long-context-attention) ——
  DSA / QSA / MSA / CSA / HCA 五种机制的独立图解讲义，含交互实验与预算计算器

## 计划中的笔记

- [ ] 推理引擎横评：vLLM / SGLang / TensorRT-LLM 在同一张卡上的实际吞吐
- [ ] PagedAttention 到底省了什么：KV cache 碎片与显存占用实测
- [ ] 连续批处理（continuous batching）的调度行为与尾延迟
- [ ] 单机多卡到 K8s 集群的部署路径：从裸机 script 到 operator
- [ ] 量化上线前的验收清单：精度回归、显存、吞吐三件套
- [ ] GPU 利用率排查：nvidia-smi 显示 100% 但吞吐上不去的常见原因

## 待办

- [ ] 建一份「显存估算速查表」（参数 / 优化器状态 / KV cache / 激活值）
