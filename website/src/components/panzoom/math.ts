// Pure view maths for PanZoomViewport. A view maps content coordinates to viewport pixels:
// screen = content * scale + (x, y). Content is drawn with transform-origin 0 0.

export interface View {
  scale: number
  x: number
  y: number
}

export interface Size {
  width: number
  height: number
}

export interface Limits {
  min: number
  max: number
}

/** The view that shows the whole content, centred, with `padding` pixels on the tighter axis. */
export function fitView(content: Size, viewport: Size, padding = 0): View {
  const w = Math.max(1, viewport.width - 2 * padding)
  const h = Math.max(1, viewport.height - 2 * padding)
  const scale = Math.min(w / content.width, h / content.height)
  return {
    scale,
    x: (viewport.width - content.width * scale) / 2,
    y: (viewport.height - content.height * scale) / 2,
  }
}

/** Zoom limits: never smaller than the fitted view, never past `maxScale` (unless fit already is). */
export function limitsFor(content: Size, viewport: Size, maxScale: number, padding = 0): Limits {
  const min = fitView(content, viewport, padding).scale
  return { min, max: Math.max(min, maxScale) }
}

function clampAxis(offset: number, contentPx: number, viewportPx: number): number {
  // Smaller than the viewport: centre it. Larger: no gap may open at either edge.
  if (contentPx <= viewportPx) return (viewportPx - contentPx) / 2
  return Math.min(0, Math.max(viewportPx - contentPx, offset))
}

export function clampView(view: View, content: Size, viewport: Size, limits: Limits): View {
  const scale = Math.min(limits.max, Math.max(limits.min, view.scale))
  return {
    scale,
    x: clampAxis(view.x, content.width * scale, viewport.width),
    y: clampAxis(view.y, content.height * scale, viewport.height),
  }
}

/** Zoom by `factor` keeping the content point under viewport `point` fixed. */
export function zoomAt(view: View, factor: number, point: { x: number; y: number }, content: Size, viewport: Size, limits: Limits): View {
  const scale = Math.min(limits.max, Math.max(limits.min, view.scale * factor))
  const k = scale / view.scale
  return clampView({ scale, x: point.x - (point.x - view.x) * k, y: point.y - (point.y - view.y) * k }, content, viewport, limits)
}

export function panBy(view: View, dx: number, dy: number, content: Size, viewport: Size, limits: Limits): View {
  return clampView({ ...view, x: view.x + dx, y: view.y + dy }, content, viewport, limits)
}

/** A 1-2-5 length (in content units × unitSize) whose bar is close to `targetPx` on screen. */
export function niceScaleBar(scale: number, unitSize: number, targetPx = 120): { length: number; px: number } {
  if (!(scale > 0 && Number.isFinite(scale))) return { length: 0, px: 0 }
  const perPx = unitSize / scale
  const raw = targetPx * perPx
  const pow = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * pow).filter((v) => v <= raw).pop() ?? pow
  return { length: step, px: step / perPx }
}
