import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import sitesFile from '../../public/data/mars/sites.json'
import { App } from '../App'
import type { MarsSitesFile } from '../data/types'

const sites = (sitesFile as MarsSitesFile).sites

function Location() {
  const loc = useLocation()
  return <output data-testid="location">{loc.pathname + loc.search}</output>
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <Location />
    </MemoryRouter>,
  )
}

// jsdom has no layout; give the pan/zoom viewport a desktop-like size.
const sizes = { clientWidth: 900, clientHeight: 700 }
const originals = Object.fromEntries(
  Object.keys(sizes).map((k) => [k, Object.getOwnPropertyDescriptor(Element.prototype, k)!]),
)
beforeAll(() => {
  for (const [k, v] of Object.entries(sizes))
    Object.defineProperty(Element.prototype, k, {
      configurable: true,
      get(this: Element) {
        return this.classList.contains('panzoom-viewport') ? v : 0
      },
    })
})
afterAll(() => {
  for (const [k, d] of Object.entries(originals)) Object.defineProperty(Element.prototype, k, d)
})

const rail = () => screen.getByRole('navigation', { name: 'Sites' })
const viewer = () => screen.getByRole('region', { name: /orthoimage strip/ })
const scaleOf = () => Number(viewer().querySelector<HTMLElement>('.panzoom-content')!.dataset.scale)

describe('Mars explorer', () => {
  it('lists the five sites in site order with exported tile counts', () => {
    renderAt('/mars')
    const items = within(rail()).getAllByRole('link')
    expect(items).toHaveLength(5)
    items.forEach((item, i) => {
      expect(item).toHaveTextContent(sites[i].name)
      expect(item).toHaveTextContent(`${sites[i].n_tiles.toLocaleString('en-GB')} tiles`)
    })
    expect(document.querySelector('[data-winner]')).toBeNull()
  })

  it('defaults to the first site and marks it current', () => {
    renderAt('/mars')
    expect(within(rail()).getAllByRole('link')[0]).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(sites[0].name)
  })

  it('reads ?site= and keeps other params when switching site', () => {
    renderAt('/mars?site=site_04&model=qcnn&seed=43')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Coprates Chasma')
    fireEvent.click(within(rail()).getByRole('link', { name: /Kaiser Crater/ }))
    const search = new URLSearchParams(screen.getByTestId('location').textContent!.split('?')[1])
    expect(Object.fromEntries(search)).toEqual({ site: 'site_02', model: 'qcnn', seed: '43' })
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Kaiser Crater')
  })

  it('falls back to the first site for an unknown ?site=', () => {
    renderAt('/mars?site=site_99')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(sites[0].name)
  })

  it('moves focus along the rail with the arrow keys', () => {
    renderAt('/mars')
    const items = within(rail()).getAllByRole('link')
    items[0].focus()
    fireEvent.keyDown(items[0], { key: 'ArrowDown' })
    expect(items[1]).toHaveFocus()
    fireEvent.keyDown(items[1], { key: 'End' })
    expect(items[4]).toHaveFocus()
  })

  it('shows the strip with alt text naming the ortho product', () => {
    renderAt('/mars?site=site_02')
    expect(screen.getByRole('img', { name: /ESP_039245_1325_RED_D_01/ })).toBeInTheDocument()
  })

  it('zooms with the keyboard and the buttons, and Fit resets', () => {
    renderAt('/mars')
    const before = scaleOf()
    fireEvent.keyDown(viewer(), { key: '+' })
    expect(scaleOf()).toBeGreaterThan(before)
    fireEvent.keyDown(viewer(), { key: '0' })
    expect(scaleOf()).toBeCloseTo(before)
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(scaleOf()).toBeGreaterThan(before)
    fireEvent.click(screen.getByRole('button', { name: 'Fit' }))
    expect(scaleOf()).toBeCloseTo(before)
  })

  it('is keyboard focusable and explains its controls', () => {
    renderAt('/mars')
    expect(viewer()).toHaveAttribute('tabindex', '0')
    expect(viewer()).toHaveAccessibleDescription(/arrow keys to pan/)
  })

  it('carries the quantum caveat in the footer and links to About', () => {
    renderAt('/mars')
    expect(screen.getByText(/No quantum advantage claimed; circuits simulated/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Moon' })).toHaveAttribute('href', '/moon')
    expect(screen.getByRole('link', { name: 'Mars' })).toHaveAttribute('aria-current', 'page')
  })
})

describe('Mars explorer readout and site details drawer', () => {
  const toggle = () => screen.getByRole('button', { name: /Site details/ })
  const drawer = () => document.getElementById(toggle().getAttribute('aria-controls')!)!

  it('shows only the compact fact strip by default', () => {
    renderAt('/mars?site=site_05')
    const readout = screen.getByRole('region', { name: 'Site readout' })
    expect(readout).toHaveTextContent('Huo Hsing Vallis')
    expect(readout).toHaveTextContent('26.43° N')
    expect(readout).toHaveTextContent('66.60° E')
    expect(readout).toHaveTextContent('7,690')
    expect(toggle()).toHaveAttribute('aria-expanded', 'false')
    expect(drawer()).not.toHaveAttribute('data-open')
    expect(drawer()).toHaveAttribute('inert')
  })

  it('opens with the button, moves focus in, and Close returns focus', () => {
    renderAt('/mars')
    fireEvent.click(toggle())
    expect(toggle()).toHaveAttribute('aria-expanded', 'true')
    expect(drawer()).toHaveAttribute('data-open')
    expect(drawer()).not.toHaveAttribute('inert')
    expect(drawer()).toHaveFocus()
    expect(within(drawer()).getByText(sites[0].dtm_product_id)).toBeInTheDocument()
    within(drawer()).getByRole('button', { name: 'Close' }).focus()
    fireEvent.click(within(drawer()).getByRole('button', { name: 'Close' }))
    expect(drawer()).not.toHaveAttribute('data-open')
    expect(toggle()).toHaveFocus()
  })

  it('toggles with the i key and closes with Escape, returning focus', () => {
    renderAt('/mars')
    fireEvent.keyDown(document.body, { key: 'i' })
    expect(drawer()).toHaveAttribute('data-open')
    fireEvent.keyDown(drawer(), { key: 'Escape' })
    expect(drawer()).not.toHaveAttribute('data-open')
    expect(toggle()).toHaveFocus()
    fireEvent.keyDown(document.body, { key: 'i' })
    fireEvent.keyDown(document.body, { key: 'i' })
    expect(drawer()).not.toHaveAttribute('data-open')
  })

  it('keeps the required caveats outside the drawer', () => {
    renderAt('/mars')
    for (const text of [/No quantum advantage claimed; circuits simulated/, /^Display stretch only$/]) {
      const el = screen.getByText(text)
      expect(drawer().contains(el)).toBe(false)
    }
  })
})
