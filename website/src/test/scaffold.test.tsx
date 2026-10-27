import { render, screen } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

describe('routing', () => {
  it('renders the landing page at /', () => {
    renderAt('/')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Choose a world.')
  })

  it.each(['mars', 'moon'] as const)('applies the %s world tokens', (world) => {
    const { container } = renderAt(`/${world}`)
    expect(container.querySelector(`[data-world="${world}"]`)).not.toBeNull()
    expect(document.documentElement.dataset.world).toBe(world)
  })

  it('keeps About and the landing page on the neutral shell', () => {
    const { container } = renderAt('/about')
    expect(container.querySelector('[data-world]')).toBeNull()
    expect(document.documentElement.dataset.world).toBeUndefined()
  })

  it('redirects unknown paths to the landing page', () => {
    renderAt('/nowhere')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Choose a world.')
  })
})

describe('theme tokens', () => {
  const css = readFileSync(resolve(import.meta.dirname, '../styles/tokens.css'), 'utf8')

  function block(selector: string): string {
    const start = css.indexOf(selector)
    expect(start, `missing ${selector}`).toBeGreaterThanOrEqual(0)
    return css.slice(start, css.indexOf('}', start))
  }

  function hexOf(body: string, token: string): string {
    const m = body.match(new RegExp(`${token}:\\s*(#[0-9a-fA-F]{6})`))
    expect(m, `missing ${token}`).not.toBeNull()
    return m![1].toLowerCase()
  }

  const required = [
    '--bg', '--surface', '--surface-2', '--border', '--text', '--text-muted',
    '--accent', '--accent-2', '--model-cnn', '--model-qcnn',
    '--truth-safe', '--truth-unsafe', '--focus',
  ]

  it.each(["[data-world='mars']", "[data-world='moon']"])('%s defines every token', (sel) => {
    const body = block(sel)
    for (const token of required) hexOf(body, token)
  })

  it('defines the MLP colour for the Moon', () => {
    hexOf(block("[data-world='moon']"), '--model-mlp')
  })

  // CNN is always the warm hue and the hybrid/QCNN always the cool hue.
  function hue(hex: string): number {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    const max = Math.max(r, g, b)
    const d = max - Math.min(r, g, b)
    if (d === 0) return 0
    const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4
    return (h * 60 + 360) % 360
  }

  it.each(["[data-world='mars']", "[data-world='moon']"])('%s: CNN warm, QCNN cool', (sel) => {
    const body = block(sel)
    const cnn = hue(hexOf(body, '--model-cnn'))
    const qcnn = hue(hexOf(body, '--model-qcnn'))
    expect(cnn).toBeGreaterThanOrEqual(15)
    expect(cnn).toBeLessThanOrEqual(60)
    expect(qcnn).toBeGreaterThanOrEqual(170)
    expect(qcnn).toBeLessThanOrEqual(250)
  })
})
