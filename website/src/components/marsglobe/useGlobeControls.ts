import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { baseRadius, clampView, nearestAngle, type GlobeView, type Size } from './projection'

const FLY_MS = 1600
const INERTIA_DECAY = 4.5 // per second: velocity × e^(−decay·t)
const MIN_SPEED = 0.002 // rad/s below which inertia stops
const AUTO_SPEED = (2 * Math.PI) / 240 // one turn in four minutes
const AUTO_IDLE_MS = 5000 // auto-rotate resumes this long after the last interaction
const KEY_TURN = (12 * Math.PI) / 180
const KEY_ZOOM = 1.5

interface Options {
  initial: GlobeView
  size: Size
  /** Reduced motion: no fly-to, no inertia, no auto-rotate. */
  reduced: boolean
  /** Ambient spin when idle (also off while a site is selected or motion is paused). */
  autoRotate: boolean
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2)

/**
 * View state and input for the globe: drag to rotate, wheel or pinch to zoom, arrow and +/− keys,
 * damped inertia after a flick, fly-to for a selected site, and a slow idle spin.
 */
export function useGlobeControls({ initial, size, reduced, autoRotate }: Options) {
  const [view, setViewState] = useState(initial)
  const current = useRef(initial)
  const velocity = useRef({ yaw: 0, pitch: 0 })
  const tween = useRef<{ from: GlobeView; to: GlobeView; start: number } | null>(null)
  const lastInput = useRef(-Infinity)
  const frame = useRef(0)
  const opts = useRef({ size, reduced, autoRotate })
  useLayoutEffect(() => {
    opts.current = { size, reduced, autoRotate }
  }, [size, reduced, autoRotate])

  const commit = useCallback((v: GlobeView) => {
    current.current = clampView(v)
    setViewState(current.current)
  }, [])

  const tick = useCallback(
    function step(now: number, prev: number) {
      const dt = Math.min(0.05, (now - prev) / 1000)
      const { reduced, autoRotate } = opts.current
      let v = current.current
      let active = false
      if (tween.current) {
        const t = Math.min(1, (now - tween.current.start) / FLY_MS)
        const { from, to } = tween.current
        const k = ease(t)
        // Zoom eases in log space so the approach feels even at every scale.
        v = {
          yaw: from.yaw + (to.yaw - from.yaw) * k,
          pitch: from.pitch + (to.pitch - from.pitch) * k,
          zoom: Math.exp(Math.log(from.zoom) + (Math.log(to.zoom) - Math.log(from.zoom)) * k),
        }
        if (t >= 1) tween.current = null
        active = true
      } else if (!reduced && Math.hypot(velocity.current.yaw, velocity.current.pitch) > MIN_SPEED) {
        v = { ...v, yaw: v.yaw + velocity.current.yaw * dt, pitch: v.pitch + velocity.current.pitch * dt }
        const decay = Math.exp(-INERTIA_DECAY * dt)
        velocity.current = { yaw: velocity.current.yaw * decay, pitch: velocity.current.pitch * decay }
        active = true
      } else if (!reduced && autoRotate && now - lastInput.current > AUTO_IDLE_MS) {
        v = { ...v, yaw: v.yaw + AUTO_SPEED * dt }
        active = true
      }
      if (v !== current.current) commit(v)
      frame.current = active || (autoRotate && !reduced) ? requestAnimationFrame((n) => step(n, now)) : 0
    },
    [commit],
  )

  const wake = useCallback(() => {
    if (!frame.current && typeof requestAnimationFrame === 'function') {
      const start = performance.now()
      frame.current = requestAnimationFrame((n) => tick(n, start))
    }
  }, [tick])

  useEffect(() => {
    wake()
    return () => {
      cancelAnimationFrame(frame.current)
      frame.current = 0
    }
  }, [wake, autoRotate, reduced])

  /** Fly to a view; under reduced motion, jump. */
  const flyTo = useCallback(
    (to: GlobeView) => {
      velocity.current = { yaw: 0, pitch: 0 }
      lastInput.current = performance.now()
      const from = current.current
      const target = clampView({ ...to, yaw: nearestAngle(from.yaw, to.yaw) })
      if (opts.current.reduced || typeof requestAnimationFrame !== 'function') {
        tween.current = null
        commit(target)
        return
      }
      tween.current = { from, to: target, start: performance.now() }
      wake()
    },
    [commit, wake],
  )

  const nudge = useCallback(
    (dYaw: number, dPitch: number, zoomBy = 1) => {
      tween.current = null
      velocity.current = { yaw: 0, pitch: 0 }
      lastInput.current = performance.now()
      const v = current.current
      commit({ yaw: v.yaw + dYaw, pitch: v.pitch + dPitch, zoom: v.zoom * zoomBy })
      wake()
    },
    [commit, wake],
  )

  // ---------------------------------------------------------------- pointer, wheel, keys

  const pointers = useRef(new Map<number, { x: number; y: number; t: number }>())

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    // A press on a pin or callout is a click, not a drag: capturing it would steal the click.
    if (e.target instanceof Element && e.target.closest('button, a')) return
    e.currentTarget.setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp })
    tween.current = null
    velocity.current = { yaw: 0, pitch: 0 }
    lastInput.current = performance.now()
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const map = pointers.current
    const prev = map.get(e.pointerId)
    if (!prev) return
    const now = { x: e.clientX, y: e.clientY, t: e.timeStamp }
    lastInput.current = performance.now()
    if (map.size === 1) {
      // The surface under the pointer follows it: one pixel turns 1 / (radius on screen) radians.
      const k = 1 / (baseRadius(opts.current.size) * current.current.zoom)
      const dYaw = (now.x - prev.x) * k
      const dPitch = (now.y - prev.y) * k
      const dt = Math.max(1, now.t - prev.t) / 1000
      velocity.current = { yaw: dYaw / dt, pitch: dPitch / dt }
      const v = current.current
      commit({ ...v, yaw: v.yaw + dYaw, pitch: v.pitch + dPitch })
    } else if (map.size === 2) {
      const other = [...map.entries()].find(([id]) => id !== e.pointerId)![1]
      const before = Math.hypot(prev.x - other.x, prev.y - other.y)
      const after = Math.hypot(now.x - other.x, now.y - other.y)
      if (before > 0) commit({ ...current.current, zoom: current.current.zoom * (after / before) })
      velocity.current = { yaw: 0, pitch: 0 }
    }
    map.set(e.pointerId, now)
  }

  const onPointerEnd = (e: React.PointerEvent) => {
    const p = pointers.current.get(e.pointerId)
    pointers.current.delete(e.pointerId)
    // A release after the pointer rested is not a flick.
    if (p && e.timeStamp - p.t > 80) velocity.current = { yaw: 0, pitch: 0 }
    if (opts.current.reduced) velocity.current = { yaw: 0, pitch: 0 }
    wake()
  }

  const onWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault()
      const lines = e.deltaMode === 1 ? 16 : 1
      nudge(0, 0, Math.exp(-e.deltaY * lines * 0.0015))
    },
    [nudge],
  )

  const onKeyDown = (e: React.KeyboardEvent): boolean => {
    if (e.altKey || e.ctrlKey || e.metaKey) return false
    const turn = KEY_TURN / Math.sqrt(current.current.zoom)
    const moves: Record<string, () => void> = {
      ArrowLeft: () => nudge(turn, 0),
      ArrowRight: () => nudge(-turn, 0),
      ArrowUp: () => nudge(0, turn),
      ArrowDown: () => nudge(0, -turn),
      '+': () => nudge(0, 0, KEY_ZOOM),
      '=': () => nudge(0, 0, KEY_ZOOM),
      '-': () => nudge(0, 0, 1 / KEY_ZOOM),
      _: () => nudge(0, 0, 1 / KEY_ZOOM),
    }
    const move = moves[e.key]
    if (!move) return false
    e.preventDefault()
    move()
    return true
  }

  return {
    view,
    flyTo,
    onWheel,
    onKeyDown,
    handlers: { onPointerDown, onPointerMove, onPointerUp: onPointerEnd, onPointerCancel: onPointerEnd },
  }
}
