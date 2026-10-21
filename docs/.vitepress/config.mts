import { defineConfig } from 'vitepress'

// 站点部署在 GitHub Pages 项目站点上：https://tju-yxq.github.io/wiki/
// 改仓库名或换成自定义域名时，只需要改这一处。
const base = '/wiki/'
const siteUrl = 'https://tju-yxq.github.io'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: "xdz' wiki",
  description: 'LLM Safety · Multi-Agent System · AI Infra —— 学习笔记与工程记录',
  cleanUrls: false,
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: `${base}images/favicon-32x32.png` }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: `${base}images/apple-touch-icon.png` }],
    ['meta', { name: 'theme-color', content: '#3c8772' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }]
  ],

  sitemap: {
    // VitePress 用 new URL(item.url, hostname) 拼 sitemap，而 url 是 "/about.html"
    // 这样的绝对路径，会直接把 base 吃掉。所以：hostname 带尾斜杠 + 把 url 转成相对。
    hostname: `${siteUrl}/wiki/`,
    transformItems: (items) =>
      items.map((item) => ({ ...item, url: String(item.url).replace(/^\//, '') }))
  },

  markdown: {
    // 数学公式：由 markdown-it-mathjax3 提供，$...$ 行内、$$...$$ 块级
    math: true,
    lineNumbers: true
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '笔记', link: '/notes/' },
      { text: '使用说明', link: '/guide/' },
      { text: '关于', link: '/about' },
      { text: '学术主页', link: 'https://tju-yxq.github.io/homepage/' }
    ],

    sidebar: {
      '/notes/': [
        {
          text: '笔记总览',
          items: [{ text: '全部笔记', link: '/notes/' }]
        },
        {
          text: 'LLM Safety',
          collapsed: false,
          items: [{ text: '本类说明', link: '/notes/llm-safety/' }]
        },
        {
          text: 'Multi-Agent System',
          collapsed: false,
          items: [{ text: '本类说明', link: '/notes/multi-agent/' }]
        },
        {
          text: 'AI Infra',
          collapsed: false,
          items: [{ text: '本类说明', link: '/notes/ai-infra/' }]
        },
        {
          text: 'Cybersecurity',
          collapsed: false,
          items: [{ text: '本类说明', link: '/notes/security/' }]
        },
        {
          text: 'Engineering',
          collapsed: false,
          items: [{ text: '本类说明', link: '/notes/engineering/' }]
        }
      ],
      '/guide/': [
        {
          text: '使用说明',
          items: [{ text: '写作与部署', link: '/guide/' }]
        }
      ]
    },

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新于', formatOptions: { dateStyle: 'short', timeStyle: 'short' } },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    langMenuLabel: '语言',

    search: {
      provider: 'local',
      options: {
        // VitePress 默认的 minisearch 分词对中文无效：整段中文会被当成一个 token，
        // 于是搜「鲁棒」找不到「多轮对话中的越狱鲁棒性」。这里用 Intl.Segmenter
        // 做中文分词（无该 API 的浏览器退化为二元组），构建端和查询端共用同一份
        // 实现 —— VitePress 会把此函数序列化进客户端 site data，注意必须自包含，
        // 不能引用模块作用域的变量。
        miniSearch: {
          options: {
            tokenize: (text: string) => {
              const CJK =
                /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]/
              const tokens: string[] = []
              const seg =
                typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function'
                  ? new Intl.Segmenter('zh', { granularity: 'word' })
                  : null
              const parts = String(text)
                .split(
                  /([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+)/
                )
                .filter(Boolean)
              for (const part of parts) {
                if (CJK.test(part) && !/[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]/.test(part)) {
                  if (seg) {
                    for (const s of seg.segment(part)) {
                      if (s.isWordLike) tokens.push(s.segment)
                    }
                  } else {
                    for (let i = 0; i < part.length; i++) tokens.push(part.slice(i, i + 2))
                  }
                } else {
                  for (const w of part.split(/[^\u0041-\u005a\u0061-\u007a\u0030-\u0039\u00c0-\u024f]+/))
                    if (w) tokens.push(w.toLowerCase())
                }
              }
              return tokens
            }
          }
        },
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜索', buttonAriaLabel: '搜索笔记' },
              modal: {
                displayDetails: '显示详情',
                resetButtonTitle: '清除查询',
                backButtonTitle: '返回',
                noResultsText: '没有找到结果',
                footer: {
                  selectText: '选择',
                  selectKeyAriaLabel: '回车',
                  navigateText: '切换',
                  navigateUpKeyAriaLabel: '上箭头',
                  navigateDownKeyAriaLabel: '下箭头',
                  closeText: '关闭',
                  closeKeyAriaLabel: 'Esc'
                }
              }
            }
          }
        }
      }
    },

    editLink: {
      pattern: 'https://github.com/tju-yxq/wiki/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/tju-yxq/wiki' }],

    footer: {
      message: '内容以 CC BY 4.0 共享，代码以 MIT 授权',
      copyright: '© 2026 喻新强 · 由 VitePress 构建'
    }
  }
})
