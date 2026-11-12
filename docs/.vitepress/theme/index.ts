import DefaultTheme from 'vitepress/theme'
import './custom.css'

const STORAGE_KEY = 'wiki-sidebar-collapsed'

function setupSidebarToggle() {
  if (document.getElementById('sb-toggle')) return

  const btn = document.createElement('button')
  btn.id = 'sb-toggle'
  btn.className = 'sb-toggle'
  btn.type = 'button'
  btn.setAttribute('aria-label', '收起或展开侧边栏')

  const apply = (collapsed: boolean) => {
    document.documentElement.classList.toggle('sb-collapsed', collapsed)
    btn.textContent = collapsed ? '»' : '«'
    btn.title = collapsed ? '展开侧边栏' : '收起侧边栏'
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0')
    } catch {}
  }

  btn.addEventListener('click', () => {
    apply(!document.documentElement.classList.contains('sb-collapsed'))
  })

  document.body.appendChild(btn)

  let saved = false
  try {
    saved = localStorage.getItem(STORAGE_KEY) === '1'
  } catch {}
  apply(saved)
}

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', setupSidebarToggle)
    } else {
      setupSidebarToggle()
    }
    // SPA 路由切换后按钮可能被移除，重新挂一次
    window.addEventListener('vitepress:routeChanged', () => setTimeout(setupSidebarToggle, 0))
  }
}
