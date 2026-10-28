// Accessibility pass (step 12): live-region announcements on site, class, model and seed changes,
// and a table fallback beside every chart.
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'

const PUBLIC = resolve(import.meta.dirname, '../../public')

const sizes = { clientWidth: 900, clientHeight: 700 }
const originals = Object.fromEntries(Object.keys(sizes).map((k) => [k, Object.getOwnPropertyDescriptor(Element.prototype, k)!]))
beforeAll(() => {
  vi.stubGlobal('fetch', async (url: string) => {
    const body = readFileSync(resolve(PUBLIC, decodeURI(String(url)).replace(/^\//, '')), 'utf8')
    return { ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body }
  })
  for (const [k, v] of Object.entries(sizes))
    Object.defineProperty(Element.prototype, k, {
      configurable: true,
      get(this: Element) {
        return this.classList.contains('panzoom-viewport') || this.classList.contains('mosaic') ? v : 0
      },
    })
})
afterAll(() => {
  vi.unstubAllGlobals()
  for (const [k, d] of Object.entries(originals)) Object.defineProperty(Element.prototype, k, d)
})
beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }))
afterEach(() => vi.useRealTimers())

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )

const announcer = () => document.querySelector('[data-announcer]') as HTMLElement
const settle = () => act(() => vi.advanceTimersByTime(100))

describe('live-region announcements', () => {
  it('Mars explorer: silent on load, then reads out a new site and a new model', async () => {
    renderAt('/mars/explore?site=site_01')
    expect(announcer()).toHaveAttribute('aria-live', 'polite')
    expect(announcer()).toHaveTextContent(/^$/)
    fireEvent.click(within(screen.getByRole('navigation', { name: 'Sites' })).getByRole('link', { name: /Kaiser Crater/ }))
    await settle()
    expect(announcer()).toHaveTextContent('Kaiser Crater, test site 2 of 5, 1,885 tiles. CNN, seed 42.')
    fireEvent.click(screen.getByRole('radio', { name: 'Hybrid CNN–QCNN' }))
    await settle()
    expect(announcer()).toHaveTextContent('Kaiser Crater, test site 2 of 5, 1,885 tiles. Hybrid CNN–QCNN, seed 42.')
  })

  it('Moon explorer: reads out a new class', async () => {
    renderAt('/moon/explore?class=ejecta')
    fireEvent.click(within(screen.getByRole('navigation', { name: 'Classes' })).getByRole('link', { name: /Old crater/ }))
    await settle()
    expect(announcer()).toHaveTextContent('Old crater, 157 test crops. CNN, seed 42.')
  })

  it('Lab: reads out a new fold', async () => {
    renderAt('/lab')
    const group = await screen.findByRole('group', { name: 'Mars fold' }, { timeout: 5000 })
    fireEvent.click(within(group).getByRole('button', { name: /Fold 1/ }))
    await settle()
    expect(announcer()).toHaveTextContent('Mars curves: fold 1, Kaiser Crater, seed 42. Moon curves: seed 42.')
  })
})

describe('table fallbacks for charts', () => {
  it('every Lab curve has a table with one row per epoch', async () => {
    renderAt('/lab')
    await screen.findByRole('group', { name: 'Mars fold' }, { timeout: 5000 })
    const charts = document.querySelectorAll('.curve')
    expect(charts.length).toBe(4)
    for (const c of charts) {
      const table = within(c as HTMLElement).getByRole('table')
      const points = c.querySelector('.curve-series')!.querySelectorAll('circle').length
      expect(within(table).getAllByRole('row').length).toBeGreaterThanOrEqual(points + 1)
    }
  })

  it('every results metric on both explorers has a table view', async () => {
    for (const path of ['/mars/explore?site=site_01', '/moon/explore?class=ejecta']) {
      const { unmount } = renderAt(path)
      await screen.findAllByText(/^Table view:/, undefined, { timeout: 5000 })
      const groups = document.querySelectorAll('.results details')
      expect(groups.length, path).toBeGreaterThan(0)
      for (const d of groups) if (/Table view/.test(d.querySelector('summary')!.textContent!)) expect(d.querySelector('table'), path).not.toBeNull()
      unmount()
    }
  })
})
