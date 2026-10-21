# xdz' wiki

个人技术笔记站，构建产物部署在 **<https://tju-yxq.github.io/wiki/>**。

基于 [VitePress](https://vitepress.dev/)：正文是纯 Markdown，构建产物是纯静态文件，
推送到 `main` 后由 GitHub Actions 自动构建并部署到 GitHub Pages。

## 本地跑起来

```bash
npm install
npm run dev        # http://localhost:5173/wiki/
```

生产构建本地验证：

```bash
npm run build      # 输出到 docs/.vitepress/dist
npm run preview
```

## 目录

```
docs/                    所有内容（只有这里会被发布）
  index.md               首页
  about.md               关于
  guide/index.md         写作与部署说明 ← 先看这个
  notes/
    index.md             笔记总览
    llm-safety/          LLM Safety
    multi-agent/         Multi-Agent System
    ai-infra/            AI Infra
    security/            网络安全
    engineering/         工程与环境
  .vitepress/config.mts  站点配置（导航、侧边栏、搜索）
  public/                静态资源与独立 HTML 页面，原样复制到产物根目录
.github/workflows/deploy.yml  自动部署
```

## 加一篇笔记

1. 在对应分类目录下新建 `.md` 文件，第一行写 `# 标题`。
2. 在 `docs/.vitepress/config.mts` 的 `sidebar` 对应分类里加一行 `{ text, link }`。
3. `git push`，等 Actions 跑完。

细节和 Markdown 可用语法见站点里的 **[写作与部署](https://tju-yxq.github.io/wiki/guide/)**。

## 两个需要注意的地方

- **`base` 必须与部署路径一致。** `docs/.vitepress/config.mts` 顶部 `const base = '/wiki/'`
  决定了所有资源和链接前缀。仓库改名、或以后换自定义域名，这里必须同步改，否则全站 404。
- **中文搜索靠自定义分词器。** VitePress 默认分词对中文无效（整段中文当一个 token，
  搜「鲁棒」找不到「越狱鲁棒性」）。配置里用 `Intl.Segmenter` 补了分词，
  且该函数会被序列化进客户端 —— 因此它**必须自包含**，不能引用模块作用域的变量。

## 相关站点

- 学术主页：<https://tju-yxq.github.io/homepage/>（仓库 `tju-yxq/homepage`，Jekyll）
- 导航页：<https://tju-yxq.github.io/>（仓库 `tju-yxq/tju-yxq.github.io`）

## 许可

笔记内容 CC BY 4.0，代码 MIT。
