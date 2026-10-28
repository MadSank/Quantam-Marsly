import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useMotion } from '../motionContext'
import { clampView, fitView, limitsFor, panBy, zoomAt, type Size, type View } from './math'

const PADDING = 16
const STEP = 1.5 // zoom factor per key press or button
const PAN_FRACTION = 0.15 // arrow keys pan by this share of the viewport (Shift: three times as far)
const TWEEN_MS = 220 // --dur-short; discrete moves (keys, buttons) ease, continuous ones (drag, wheel) do not
const TAP_SLOP = 4 // a press that moves less than this many pixels is a tap, not a drag

/**
 * The view on screen. Discrete moves ease in JavaScript rather than by a CSS transition, so
 * anything drawn from the view (the tile overlay canvas) stays aligned frame by frame.
 */
function useShownView(target: View, smooth: boolean): View {
  const [frame, setFrame] = useState<View | null>(null)
  const last = useRef(target)

  useLayoutEffect(() => {
    if (!smooth || typeof requestAnimationFrame !== 'function') {
      last.current = target
      return
    }
    const from = last.current
    const start = performance.now()
    let raf = requestAnimationFrame(function step(now) {
      const t = Math.min(1, (now - start) / TWEEN_MS)
      const k = 1 - (1 - t) ** 3
      const v = t < 1 ? { scale: from.scale + (target.scale - from.scale) * k, x: from.x + (target.x - from.x) * k, y: from.y + (target.y - from.y) * k } : target
      last.current = v
      setFrame(t < 1 ? v : null)
      if (t < 1) raf = requestAnimationFrame(step)
    })
    return () => cancelAnimationFrame(raf)
  }, [target, smooth])

  return frame ?? target
}

interface Props {
  /** Content size in content units (for Mars strips: native ortho pixels). */
  content: Size
  /** Largest scale allowed, in viewport pixels per content unit. */
  maxScale: number
  label: string
  /** Drawn inside the transformed layer, at content size. */
  children: ReactNode
  /** Drawn above the content, untransformed (overlays, scale bars, notices), from the view on screen. */
  hud?: (view: View, viewport: Size) => ReactNode
  /** Extra controls in the control row, before the zoom readout. */
  toolbar?: ReactNode
  /** A click or tap without dragging, in content coordinates. */
  onTap?: (point: { x: number; y: number }) => void
  /** Keys the host handles while the viewer has focus (e.g. Enter to inspect the centre). */
  onKey?: (e: React.KeyboardEvent, centre: { x: number; y: number }) => boolean
  className?: string
}

/**
 * Shared pan and zoom surface: wheel and buttons zoom, drag pans, two-finger pinch zooms,
 * and the keyboard does both (+/− zoom, arrows pan, 0 resets). Until the reader moves the
 * view it stays fitted to the viewport, including across resizes.
 */
export function PanZoomViewport({ content, maxScale, label, children, hud, toolbar, onTap, onKey, className }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const helpId = useId()
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 })
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 })
  const [touched, setTouched] = useState(false)
  const [smooth, setSmooth] = useState(false)
  const [announcement, setAnnouncement] = useState('')
  const { reduced } = useMotion()
  const shown = useShownView(view, smooth && !reduced)

  const limits = limitsFor(content, viewport, maxScale, PADDING)
  const ready = viewport.width > 0 && viewport.height > 0

  // Track the viewport size.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setViewport({ width: el.clientWidth, height: el.clientHeight })
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Fitted until touched; afterwards keep the reader's view, re-clamped to the new size.
  useLayoutEffect(() => {
    if (!ready) return
    setView((v) => (touched ? clampView(v, content, viewport, limits) : fitView(content, viewport, PADDING)))
    // limits derives from content/viewport/maxScale, which are listed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, viewport.width, viewport.height, content.width, content.height, maxScale, touched])

  const percent = (s: number) => `${s * 100 >= 10 ? Math.round(s * 100) : (s * 100).toFixed(1)}%`

  const apply = useCallback(
    (next: (v: View) => View, opts: { smooth: boolean; announce?: boolean }) => {
      setTouched(true)
      setSmooth(opts.smooth)
      setView((v) => {
        const n = next(v)
        if (opts.announce) setAnnouncement(`Zoom ${percent(n.scale)} of full resolution`)
        return n
      })
    },
    [],
  )

  const centre = { x: viewport.width / 2, y: viewport.height / 2 }
  const zoomBy = (factor: number) =>
    apply((v) => zoomAt(v, factor, centre, content, viewport, limits), { smooth: true, announce: true })
  const reset = () => {
    setTouched(false)
    setSmooth(true)
    setView(fitView(content, viewport, PADDING))
    setAnnouncement('View reset to fit')
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return
    const centreContent = { x: (viewport.width / 2 - view.x) / view.scale, y: (viewport.height / 2 - view.y) / view.scale }
    if (onKey?.(e, centreContent)) {
      e.preventDefault()
      return
    }
    const far = e.shiftKey ? 3 : 1
    const dx = viewport.width * PAN_FRACTION * far
    const dy = viewport.height * PAN_FRACTION * far
    const pan = (x: number, y: number) => apply((v) => panBy(v, x, y, content, viewport, limits), { smooth: true })
    switch (e.key) {
      case '+':
      case '=':
        zoomBy(STEP)
        break
      case '-':
      case '_':
        zoomBy(1 / STEP)
        break
      case '0':
        reset()
        break
      // Arrows move the view over the content, so the content moves the other way.
      case 'ArrowLeft':
        pan(dx, 0)
        break
      case 'ArrowRight':
        pan(-dx, 0)
        break
      case 'ArrowUp':
        pan(0, dy)
        break
      case 'ArrowDown':
        pan(0, -dy)
        break
      default:
        return
    }
    e.preventDefault()
  }

  // Wheel zoom needs a non-passive listener to stop the page scrolling.
  const latest = useRef({ content, viewport, limits, apply })
  latest.current = { content, viewport, limits, apply }
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const { content, viewport, limits, apply } = latest.current
      const rect = el.getBoundingClientRect()
      const lines = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? viewport.height : 1
      const factor = Math.exp(-e.deltaY * lines * 0.0015)
      const point = { x: e.clientX - rect.left, y: e.clientY - rect.top }
      apply((v) => zoomAt(v, factor, point, content, viewport, limits), { smooth: false })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Pointer drag (one pointer) and pinch (two pointers).
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const press = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  const [dragging, setDragging] = useState(false)

  const local = (e: React.PointerEvent) => {
    const rect = ref.current!.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = local(e)
    pointers.current.set(e.pointerId, p)
    press.current = pointers.current.size === 1 ? { ...p, moved: false } : null
    setDragging(true)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const map = pointers.current
    const prev = map.get(e.pointerId)
    if (!prev) return
    const now = local(e)
    if (press.current && Math.hypot(now.x - press.current.x, now.y - press.current.y) > TAP_SLOP) press.current.moved = true
    if (map.size === 1) {
      apply((v) => panBy(v, now.x - prev.x, now.y - prev.y, content, viewport, limits), { smooth: false })
    } else if (map.size === 2) {
      const other = [...map.entries()].find(([id]) => id !== e.pointerId)![1]
      const before = Math.hypot(prev.x - other.x, prev.y - other.y)
      const after = Math.hypot(now.x - other.x, now.y - other.y)
      const mid = { x: (now.x + other.x) / 2, y: (now.y + other.y) / 2 }
      const dx = mid.x - (prev.x + other.x) / 2
      const dy = mid.y - (prev.y + other.y) / 2
      if (before > 0)
        apply(
          (v) => panBy(zoomAt(v, after / before, mid, content, viewport, limits), dx, dy, content, viewport, limits),
          { smooth: false },
        )
    }
    map.set(e.pointerId, now)
  }

  const onPointerEnd = (e: React.PointerEvent) => {
    const tap = e.type === 'pointerup' && press.current && !press.current.moved && pointers.current.size === 1
    pointers.current.delete(e.pointerId)
    if (pointers.current.size === 0) setDragging(false)
    if (tap && onTap) {
      const p = local(e)
      onTap({ x: (p.x - view.x) / view.scale, y: (p.y - view.y) / view.scale })
    }
    if (pointers.current.size === 0) press.current = null
  }

  return (
    <div className={['panzoom', className].filter(Boolean).join(' ')}>
      <div
        ref={ref}
        className="panzoom-viewport"
        role="region"
        aria-roledescription="pan and zoom viewer"
        aria-label={label}
        aria-describedby={helpId}
        tabIndex={0}
        data-dragging={dragging || undefined}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onDoubleClick={(e) => {
          const point = { x: e.clientX - ref.current!.getBoundingClientRect().left, y: e.clientY - ref.current!.getBoundingClientRect().top }
          apply((v) => zoomAt(v, e.shiftKey ? 1 / 2 : 2, point, content, viewport, limits), { smooth: true, announce: true })
        }}
      >
        <div
          className="panzoom-content"
          data-scale={view.scale}
          style={{
            width: content.width,
            height: content.height,
            transform: `translate3d(${shown.x}px, ${shown.y}px, 0) scale(${shown.scale})`,
            visibility: ready ? undefined : 'hidden',
          }}
        >
          {children}
        </div>
        {ready && hud?.(shown, viewport)}
      </div>
      <div className="panzoom-controls">
        {toolbar}
        <div className="panzoom-buttons" role="group" aria-label="Zoom">
          <button type="button" onClick={() => zoomBy(STEP)} disabled={view.scale >= limits.max - 1e-9} aria-label="Zoom in">
            +
          </button>
          <button type="button" onClick={() => zoomBy(1 / STEP)} disabled={view.scale <= limits.min + 1e-9} aria-label="Zoom out">
            −
          </button>
          <button type="button" onClick={reset} disabled={!touched}>
            Fit
          </button>
        </div>
        <p className="panzoom-readout num" aria-hidden="true">
          {percent(view.scale)}
        </p>
        <p id={helpId} className="panzoom-help">
          Drag or use arrow keys to pan. Scroll, pinch or press + and − to zoom. Press 0 to fit.
        </p>
      </div>
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  )
}
