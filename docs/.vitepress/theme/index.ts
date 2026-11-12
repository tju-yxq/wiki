import DefaultTheme from 'vitepress/theme'
import './custom.css'

const STORAGE_KEY = 'wiki-sidebar-collapsed'

function applyState(collapsed: boolean) {
  document.documentElement.classList.toggle('sb-collapsed', collapsed)
  const btn = document.getElementById('sb-toggle')
  if (btn) {
    btn.textContent = collapsed ? '»' : '«'
    btn.title = collapsed ? '展开侧边栏' : '收起侧边栏'
    btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true')
  }
  try {
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0')
  } catch {}
}

function mountToggle() {
  if (document.getElementById('sb-toggle')) return

  // 放进导航栏右侧的 flex 区域，跟着导航栏排版，不会覆盖正文
  const host =
    document.querySelector('.VPNavBar .content-body') ||
    document.querySelector('.VPNavBar .content') ||
    document.querySelector('.VPNavBar .container')
  if (!host) return

  const btn = document.createElement('button')
  btn.id = 'sb-toggle'
  btn.className = 'sb-toggle'
  btn.type = 'button'
  btn.addEventListener('click', () => {
    applyState(!document.documentElement.classList.contains('sb-collapsed'))
  })
  host.insertBefore(btn, host.firstChild)

  let saved = false
  try {
    saved = localStorage.getItem(STORAGE_KEY) === '1'
  } catch {}
  applyState(saved)
}

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return
    const boot = () => {
      mountToggle()
      // SPA 路由切换后重新挂载，并沿用已保存的状态
      window.addEventListener('vitepress:routeChanged', () => {
        setTimeout(() => {
          mountToggle()
          applyState(document.documentElement.classList.contains('sb-collapsed'))
        }, 0)
      })
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot)
    } else {
      boot()
    }
  }
}
