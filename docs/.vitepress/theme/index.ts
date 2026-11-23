import DefaultTheme from 'vitepress/theme'
import './custom.css'

const STORAGE_KEY = 'wiki-sidebar-collapsed-v2'

const ICON_SIDEBAR = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>`

function hasSidebar(): boolean {
  return Boolean(document.querySelector('.VPSidebar'))
}

function getSavedCollapsed(): boolean {
  try {
    const val = localStorage.getItem(STORAGE_KEY)
    // 默认收起左侧导航栏，阅读与汇报更聚焦
    return val === null ? true : val === '1'
  } catch {
    return true
  }
}

function applySidebarState(collapsed: boolean, persist = true) {
  const sidebarPresent = hasSidebar()
  document.documentElement.classList.toggle('sb-collapsed', sidebarPresent && collapsed)

  const btn = document.getElementById('sb-toggle')
  if (btn) {
    btn.style.display = sidebarPresent ? '' : 'none'
    btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true')
    btn.title = collapsed ? '展开左侧目录（快捷键 [ ）' : '收起左侧目录（快捷键 [ ）'
    btn.innerHTML = `${ICON_SIDEBAR}<span class="sb-toggle__label">${collapsed ? '目录' : '收起'}</span>`
  }

  if (persist) {
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0')
    } catch {}
  }
}

function mountSidebarToggle() {
  let btn = document.getElementById('sb-toggle') as HTMLButtonElement | null
  if (!btn) {
    const host =
      document.querySelector('.VPNavBar .content-body') ||
      document.querySelector('.VPNavBar .content') ||
      document.querySelector('.VPNavBar .container')
    if (!host) return

    btn = document.createElement('button')
    btn.id = 'sb-toggle'
    btn.className = 'sb-toggle'
    btn.type = 'button'
    btn.addEventListener('click', () => {
      const next = !document.documentElement.classList.contains('sb-collapsed')
      applySidebarState(next, true)
    })
    host.insertBefore(btn, host.firstChild)
  }
  applySidebarState(getSavedCollapsed(), false)
}

function mountReadingProgress() {
  if (document.getElementById('reading-progress')) return
  const bar = document.createElement('div')
  bar.id = 'reading-progress'
  bar.className = 'reading-progress'
  document.body.appendChild(bar)

  const update = () => {
    if (!document.querySelector('.VPDoc')) {
      bar.style.width = '0%'
      return
    }
    const scrollTop = window.scrollY || document.documentElement.scrollTop
    const docHeight = document.documentElement.scrollHeight - window.innerHeight
    const pct = docHeight > 0 ? Math.min(100, Math.max(0, (scrollTop / docHeight) * 100)) : 0
    bar.style.width = `${pct.toFixed(1)}%`
  }

  window.addEventListener('scroll', update, { passive: true })
  window.addEventListener('resize', update, { passive: true })
  update()
}

function mountImageLightbox() {
  if (document.getElementById('img-lightbox')) return

  const overlay = document.createElement('div')
  overlay.id = 'img-lightbox'
  overlay.className = 'img-lightbox'
  overlay.setAttribute('role', 'dialog')
  overlay.setAttribute('aria-modal', 'true')

  const img = document.createElement('img')
  img.className = 'img-lightbox__img'
  overlay.appendChild(img)

  const close = () => {
    overlay.classList.remove('is-open')
    document.body.style.overflow = ''
  }

  overlay.addEventListener('click', close)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('is-open')) {
      close()
    }
  })

  document.body.appendChild(overlay)

  document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement | null
    if (!target || target.tagName !== 'IMG') return
    if (!target.closest('.vp-doc')) return
    const src = (target as HTMLImageElement).currentSrc || (target as HTMLImageElement).src
    if (!src) return
    img.src = src
    img.alt = (target as HTMLImageElement).alt || ''
    overlay.classList.add('is-open')
    document.body.style.overflow = 'hidden'
  })
}

function bindShortcut() {
  if ((window as any).__sbShortcutBound) return
  ;(window as any).__sbShortcutBound = true
  document.addEventListener('keydown', (e) => {
    if (e.key !== '[' || e.metaKey || e.ctrlKey || e.altKey) return
    const tag = (e.target as HTMLElement | null)?.tagName
    if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement | null)?.isContentEditable) {
      return
    }
    if (!hasSidebar()) return
    const next = !document.documentElement.classList.contains('sb-collapsed')
    applySidebarState(next, true)
  })
}

export default {
  extends: DefaultTheme,
  enhanceApp({ router }: { router: any }) {
    if (typeof window === 'undefined') return

    const syncUI = () => {
      requestAnimationFrame(() => {
        mountSidebarToggle()
        mountReadingProgress()
        mountImageLightbox()
        bindShortcut()
      })
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', syncUI)
    } else {
      syncUI()
    }

    if (router) {
      const prevChange = router.onAfterRouteChange
      router.onAfterRouteChange = (to: string) => {
        if (typeof prevChange === 'function') prevChange(to)
        setTimeout(syncUI, 30)
      }
      const prevChanged = router.onAfterRouteChanged
      router.onAfterRouteChanged = (to: string) => {
        if (typeof prevChanged === 'function') prevChanged(to)
        setTimeout(syncUI, 30)
      }
    }
  }
}
