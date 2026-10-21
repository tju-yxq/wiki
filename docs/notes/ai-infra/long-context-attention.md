# 长上下文注意力：存与读的两本账

一句话结论：**少存（压缩长期记忆的条目）和少算（稀疏每次读取的位置）是两个独立的优化轴**，DSA / QSA / MSA / CSA / HCA 的真正分歧就在这里。

这是一份**独立可交互的中文图解讲义**，不是普通的 Markdown 笔记——里面嵌了六张直接从技术报告裁下来的架构原图、一个可切换机制的注意力实验、一个百万上下文预算计算器，还有一节自测题。约 18 分钟读完。

<a class="lc-entry" href="/wiki/lectures/long-context-attention/">
  <span class="lc-entry__title">打开交互讲义 →</span>
  <span class="lc-entry__meta">18 分钟 · 6 张报告原图 · 单文件离线可用 · 支持打印为 PDF</span>
</a>

<style>
.lc-entry {
  display: block;
  margin: 28px 0;
  padding: 20px 22px;
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 8px;
  text-decoration: none;
  transition: border-color .15s ease, background-color .15s ease;
}
.lc-entry:hover { background-color: var(--vp-c-brand-soft); }
.lc-entry__title { display: block; font-size: 1.0625rem; font-weight: 600; color: var(--vp-c-brand-1); }
.lc-entry__meta { display: block; margin-top: 6px; font-size: 0.875rem; color: var(--vp-c-text-2); }
</style>

## 讲义覆盖的五个机制

| 机制 | 索引/选择单位 | 主分支读什么 | 缩短了长期主 KV？ |
| --- | --- | --- | --- |
| **DSA**（DeepSeek-V3.2） | 单个 token | 所选原始位置表示 | 否，MLA 另管特征 |
| **QSA**（Qwen3.8-Flash-Next） | 4-token 微块，池化 key | 展开后的原始 KV | 否，缩短的是索引 |
| **MSA**（MiniMax） | 各 GQA 组独立，128-token 块 | 所选块的原始 KV | 否 |
| **CSA**（DeepSeek-V4） | 压缩条目 | 所选摘要 + 局部窗口 | 是，约 N/4 |
| **HCA**（DeepSeek-V4） | 不选择 | 全部摘要 + 局部窗口 | 是，约 N/128 |

## 为什么值得单独做成一个页面

这三件事用 Markdown 讲不清楚：

1. **原图要看细节。** 六张架构图是从报告 PDF 里裁的，讲义里点一下能放大，旁边还标了「读图路线」告诉你先看哪条分支。
2. **稀疏性要看得见。** 同一段 64 token 历史，用五种机制分别演示哪些位置被选中、哪些被摘要覆盖、哪些没读——颜色一换就懂了。
3. **预算要能拖。** 把历史长度从 16K 拖到 1M，看每种机制的主分支读取量怎么变、谁先封顶。

## 一并部署的材料

- 5 份原始技术报告 PDF（讲义里的「本地 PDF」链接指向它们，与讲义同目录）
- 配套中文讲解稿：<a href="/wiki/lectures/long-context-attention/attention-explainer-script.md">attention-explainer-script.md</a>

::: tip 说明
上面这个讲解稿是 `docs/public/` 下的静态文件，不是站内页面，所以用原始 HTML 链接引用。
:::

::: tip 内容归属
讲义里的架构原图直接裁自各报告，版权归原作者；中文说明、交互色块与预算表属原创整理。参数以发布配置和报告为准，交互演示为教学缩比，不是实验复现。
:::
