# 写作与部署

这个站点用 [VitePress](https://vitepress.dev/) 构建：正文是纯 Markdown，构建产物是纯静态文件，推送到 `main` 分支后由 GitHub Actions 自动部署到 <https://tju-yxq.github.io/wiki/>。

## 站点结构

```
tju-yxq.github.io/homepage/   学术主页（Jekyll，仓库 tju-yxq/homepage）
tju-yxq.github.io/wiki/       本笔记站（VitePress，仓库 tju-yxq/wiki）
tju-yxq.github.io/            导航页（仓库 tju-yxq/tju-yxq.github.io）
```

三个站点同源、互不干扰，各自独立构建部署。任一站点挂了不影响另外两个。

## 加一篇笔记

**第一步**，在对应分类目录下新建 Markdown 文件，例如 `docs/notes/security/my-note.md`：

```markdown
# 笔记标题（标题即结论，别写「关于 XXX 的一些思考」）

一句话说清这篇要解决什么问题。

## 结论先行

能给结论就先给结论，读者不一定读完。

## 正文

推导、复现步骤、代码。代码块标注语言，会做语法高亮和行号：

````markdown
```python
def exploit(target: str) -> bool:
    return True
```
````

**第二步**，把这一页挂到左侧目录。编辑 `docs/.vitepress/config.mts`，在 `sidebar` 里对应分类下加一行：

```ts
{
  text: '网络安全',
  collapsed: false,
  items: [
    { text: '本类说明', link: '/notes/security/' },
    { text: '笔记标题', link: '/notes/security/my-note' }   // 新增这行
  ]
}
```

**第三步**，提交推送，等 Actions 跑完（约一分钟）：

```bash
git add -A
git commit -m "notes: 新增 XXX 笔记"
git push
```

::: tip 不想手动维护侧边栏？
也可以只在自己笔记的索引页里加链接。但侧边栏能保证任何一页的读者都看得到全站结构，建议加。
:::

## 本地预览

```bash
npm install        # 首次
npm run dev        # 打开 http://localhost:5173/wiki/
```

改文件即时热更新。想验证生产构建就 `npm run build && npm run preview`。

## 能用的 Markdown 扩展

除了标准 GFM（表格、任务列表、删除线），还开了：

| 写法 | 效果 |
| --- | --- |
| `$E = mc^2$` | 行内数学公式（MathJax） |
| `$$ \sum_{i=1}^n i $$` | 块级公式，独占一行居中 |
| `::: tip` / `::: warning` / `::: danger` | 提示框，见 [VitePress 文档](https://vitepress.dev/zh/guide/markdown#custom-containers) |
| `` ```python{2} `` | 代码块第 2 行高亮 |
| `[链接](./other.md)` | 相对路径引用其他笔记，构建时自动改写 |

更多语法（代码组、导入代码片段、徽章）见
[VitePress Markdown 扩展](https://vitepress.dev/zh/guide/markdown)。

## 部署细节

- 工作流：`.github/workflows/deploy.yml`，`push` 到 `main` 触发。
- `base: '/wiki/'` 写在 `docs/.vitepress/config.mts` 顶部，**如果仓库改名或换自定义域名，这里必须同步改**，否则所有资源和链接会 404。
- 构建产物在 `docs/.vitepress/dist/`，已加入 `.gitignore`，不进版本库。
- 想回滚：在 GitHub 的 Actions 页面重新运行任意一次成功的历史部署即可。

## 想换主题或组件？

VitePress 支持自定义主题（`.vitepress/theme/`）与 Vue 组件直接嵌入 Markdown。
如果只是改配色，优先覆写 CSS 变量，别 fork 主题：

```css
/* docs/.vitepress/theme/custom.css */
:root {
  --vp-c-brand-1: #3c8772;
}
```

然后在 `config.mts` 里 `import './theme/custom.css'`。
