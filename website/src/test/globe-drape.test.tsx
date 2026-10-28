import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import sitesFile from '../../public/data/mars/sites.json'
import { Callouts } from '../components/marsglobe/Callouts'
import { baseRadius, projectOrtho, viewFor, zoomForFootprint } from '../components/marsglobe/projection'
import type { MarsSitesFile } from '../data/types'

const sites = (sitesFile as MarsSitesFile).sites
const size = { width: 1600, height: 900 }

describe('close-up framing and the draped strip', () => {
  it.each(sites.map((s) => [s.site_id, s] as const))('%s: the fly-to zoom frames the strip at about 40% of the stage height', (_, s) => {
    const zoom = zoomForFootprint(s.footprint)
    const view = viewFor(s.center.lat, s.center.lon_e, zoom)
    const pts = s.footprint.map(([lat, lon]) => projectOrtho(lat, lon, view, size))
    const span = Math.max(Math.max(...pts.map((p) => p.y)) - Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.x)) - Math.min(...pts.map((p) => p.x)))
    expect(span / size.height).toBeGreaterThan(0.3)
    expect(span / size.height).toBeLessThan(0.5)
    expect(baseRadius(size) * zoom).toBeGreaterThan(0)
  })

  it.each(sites.map((s) => [s.site_id, s] as const))('%s: the preview is placed with its corners on the projected image corners', (_, s) => {
    const view = viewFor(s.center.lat, s.center.lon_e, zoomForFootprint(s.footprint))
    const project = (lat: number, lon: number) => projectOrtho(lat, lon, view, size)
    const { container } = render(<Callouts sites={sites} project={project} selected={s.site_id} footprints onSelect={() => {}} />)
    const img = container.querySelector<HTMLImageElement>(`img.drape[data-site="${s.site_id}"]`)!
    const m = img.style.transform.match(/matrix\(([^)]+)\)/)![1].split(',').map(Number)
    const apply = (x: number, y: number) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
    const { width: w, height: h } = s.preview!
    for (const [key, [x, y]] of [['top_left', [0, 0]], ['top_right', [w, 0]], ['bottom_left', [0, h]]] as const) {
      const want = project(...s.image_corners[key])
      const [gx, gy] = apply(x, y)
      expect(gx).toBeCloseTo(want.x, 1)
      expect(gy).toBeCloseTo(want.y, 1)
    }
    // The site's centre pin falls inside the draped image.
    const c = project(s.center.lat, s.center.lon_e)
    const xs = (['top_left', 'top_right', 'bottom_left'] as const).map((k) => project(...s.image_corners[k]).x)
    expect(c.x).toBeGreaterThan(Math.min(...xs))
    expect(c.x).toBeLessThan(Math.max(...xs) + (Math.max(...xs) - Math.min(...xs)))
  })

  it('draws no drape or footprint at overview zoom', () => {
    const project = (lat: number, lon: number) => projectOrtho(lat, lon, viewFor(10, 20, 1), size)
    const { container } = render(<Callouts sites={sites} project={project} selected="site_01" footprints={false} onSelect={() => {}} />)
    expect(container.querySelector('img.drape, polygon.footprint')).toBeNull()
  })
})

describe('callout labels keep out of the panels', () => {
  const project = () => ({ x: 400, y: 300, z: 1, visible: true })
  const one = sites.filter((s) => s.site_id === 'site_04') // its label points left by default

  it('points the label to its default side when nothing is in the way', () => {
    const { container } = render(<Callouts sites={one} project={project} selected={null} footprints={false} onSelect={() => {}} />)
    expect(container.querySelector('.callout')).toHaveAttribute('data-side', 'left')
  })

  it('flips the label to the other side of its pin when a panel covers the default side', () => {
    const panel = { left: 0, top: 0, right: 330, bottom: 600 }
    const { container } = render(<Callouts sites={one} project={project} selected={null} footprints={false} avoid={[panel]} onSelect={() => {}} />)
    expect(container.querySelector('.callout')).toHaveAttribute('data-side', 'right')
  })
})
