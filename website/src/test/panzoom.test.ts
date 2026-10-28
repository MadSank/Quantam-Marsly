import { describe, expect, it } from 'vitest'
import { clampView, fitView, limitsFor, niceScaleBar, panBy, zoomAt } from '../components/panzoom/math'

const content = { width: 1000, height: 4000 }
const viewport = { width: 800, height: 600 }
const limits = limitsFor(content, viewport, 2)

describe('pan/zoom maths', () => {
  it('fits the whole content, centred', () => {
    const v = fitView(content, viewport)
    expect(v.scale).toBeCloseTo(600 / 4000)
    expect(v.x).toBeCloseTo((800 - 1000 * v.scale) / 2)
    expect(v.y).toBeCloseTo(0)
  })

  it('never zooms out past the fitted view or in past the maximum', () => {
    const fit = fitView(content, viewport)
    expect(zoomAt(fit, 0.1, { x: 400, y: 300 }, content, viewport, limits).scale).toBeCloseTo(fit.scale)
    expect(zoomAt(fit, 1000, { x: 400, y: 300 }, content, viewport, limits).scale).toBe(2)
  })

  it('keeps the content point under the cursor fixed while zooming', () => {
    const start = clampView({ scale: 1, x: -100, y: -500 }, content, viewport, limits)
    const point = { x: 300, y: 200 }
    const under = { x: (point.x - start.x) / start.scale, y: (point.y - start.y) / start.scale }
    const next = zoomAt(start, 1.5, point, content, viewport, limits)
    expect(under.x * next.scale + next.x).toBeCloseTo(point.x)
    expect(under.y * next.scale + next.y).toBeCloseTo(point.y)
  })

  it('opens no gap at the edges when panning a zoomed view', () => {
    const v = { scale: 1, x: 0, y: 0 }
    expect(panBy(v, 500, 500, content, viewport, limits)).toMatchObject({ x: 0, y: 0 })
    expect(panBy(v, -5000, -9000, content, viewport, limits)).toMatchObject({ x: 800 - 1000, y: 600 - 4000 })
  })

  it('centres an axis that is smaller than the viewport', () => {
    const v = clampView({ scale: 0.15, x: 999, y: 0 }, content, viewport, limits)
    expect(v.x).toBeCloseTo((800 - 150) / 2)
  })

  it('picks a 1-2-5 scale bar length close to the target width', () => {
    const bar = niceScaleBar(0.25, 1.0, 120) // 4 m per screen pixel → 480 m target
    expect(bar.length).toBe(200)
    expect(bar.px).toBeCloseTo(50)
    expect([1, 2, 5]).toContain(bar.length / 10 ** Math.floor(Math.log10(bar.length)))
  })
})
