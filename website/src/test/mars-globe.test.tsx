// /mars globe entry view (PLAN.md step 8b). jsdom has no WebGL, so these tests exercise the
// flat-map fallback; the globe's own projection is tested in globe-projection.test.ts.
import { fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter, useLocation } from 'react-router'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import caveatsFile from '../../public/data/caveats.json'
import sitesFile from '../../public/data/mars/sites.json'
import { App } from '../App'
import { latLabel, lonLabel, siteDescription } from '../components/globeview/format'
import { flatMapRect, projectEquirect } from '../components/globeview/projection'
import type { CaveatsFile, MarsSitesFile } from '../data/types'

const PUBLIC = resolve(import.meta.dirname, '../../public')
const sites = (sitesFile as MarsSitesFile).sites
const STAGE = { width: 1200, height: 700 }

const sizes = { clientWidth: STAGE.width, clientHeight: STAGE.height }
const originals = Object.fromEntries(Object.keys(sizes).map((k) => [k, Object.getOwnPropertyDescriptor(Element.prototype, k)!]))
beforeAll(() => {
  vi.stubGlobal('fetch', async (url: string) => {
    const path = resolve(PUBLIC, decodeURI(String(url)).replace(/^\//, ''))
    return { ok: true, status: 200, json: async () => JSON.parse(readFileSync(path, 'utf8')) }
  })
  for (const [k, v] of Object.entries(sizes))
    Object.defineProperty(Element.prototype, k, {
      configurable: true,
      get(this: Element) {
        return this.classList.contains('globe-stage') || this.classList.contains('panzoom-viewport') ? v : 0
      },
    })
})
afterAll(() => {
  vi.unstubAllGlobals()
  for (const [k, d] of Object.entries(originals)) Object.defineProperty(Element.prototype, k, d)
})

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

const location = () => screen.getByTestId('location').textContent!
const list = () => screen.getByRole('navigation', { name: 'Test sites' })
// The card exists only while a site is chosen.
const card = () => screen.getByRole('complementary', { name: 'Selected site' })
const noCard = () => expect(screen.queryByRole('complementary', { name: 'Selected site' })).toBeNull()

describe('Mars globe view: fallback without WebGL', () => {
  it('opens /mars on the flat map, says why, and disables the globe toggle', () => {
    renderAt('/mars')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Mars test sites')
    expect(document.querySelector('.globe-page')).toHaveAttribute('data-mode', 'map')
    expect(document.querySelector('img.flat-map')).not.toBeNull()
    expect(screen.getByText(/cannot draw the 3D globe/)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Globe' })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'Flat map' })).toHaveAttribute('aria-checked', 'true')
  })

  it('puts every pin at its site centre on the equirectangular map', () => {
    renderAt('/mars')
    const rect = flatMapRect(STAGE)
    for (const s of sites) {
      const callout = document.querySelector<HTMLElement>(`.callout[data-site="${s.site_id}"]`)!
      expect(Number(callout.dataset.lat)).toBe(s.center.lat)
      expect(Number(callout.dataset.lon)).toBe(s.center.lon_e)
      const p = projectEquirect(s.center.lat, s.center.lon_e, rect)
      expect(Number(callout.dataset.x)).toBeCloseTo(p.x, 1)
      expect(Number(callout.dataset.y)).toBeCloseTo(p.y, 1)
    }
    // site_04 lies at 300.97° E, i.e. 59.03° W: left of the map's centre.
    const coprates = document.querySelector<HTMLElement>('.callout[data-site="site_04"]')!
    expect(Number(coprates.dataset.x)).toBeLessThan(rect.x + rect.width / 2)
  })
})

describe('Mars globe view: site list and card', () => {
  it('lists the five sites in order with their coordinates', () => {
    renderAt('/mars')
    const items = within(list()).getAllByRole('button')
    expect(items.map((b) => b.querySelector('.site-name')!.textContent)).toEqual(sites.map((s) => s.name))
    items.forEach((b, i) => expect(b).toHaveTextContent(`${latLabel(sites[i].center.lat)} ${lonLabel(sites[i].center.lon_e)}`))
  })

  it('moves along the list with the arrow keys', () => {
    renderAt('/mars')
    const items = within(list()).getAllByRole('button')
    items[0].focus()
    fireEvent.keyDown(items[0], { key: 'ArrowDown' })
    expect(items[1]).toHaveFocus()
    fireEvent.keyDown(items[1], { key: 'End' })
    expect(items[4]).toHaveFocus()
  })

  it.each(sites.map((s) => s.site_id))('%s: the card shows sites.json content and links to the explorer', (id) => {
    const s = sites.find((x) => x.site_id === id)!
    renderAt('/mars')
    noCard()
    fireEvent.click(within(list()).getByRole('button', { name: new RegExp(s.name) }))
    expect(location()).toBe(`/mars?site=${id}`)
    expect(within(card()).getByRole('heading', { level: 2 })).toHaveTextContent(s.name)
    expect(card()).toHaveTextContent(`${latLabel(s.center.lat)} ${lonLabel(s.center.lon_e)}`)
    expect(card()).toHaveTextContent(`${s.n_tiles.toLocaleString('en-GB')} tiles`)
    expect(card()).toHaveTextContent(siteDescription(s))
    expect(within(card()).getByRole('img')).toHaveAttribute('alt', expect.stringContaining(s.ortho_product_id))
    expect(within(card()).getByRole('link', { name: 'Explore site' })).toHaveAttribute('href', `/mars/explore?site=${id}`)
    expect(within(list()).getByRole('button', { name: new RegExp(s.name) })).toHaveAttribute('aria-current', 'true')
  })

  it('a pin opens the same card', () => {
    renderAt('/mars')
    fireEvent.click(document.querySelector('.pin-hit[data-site="site_03"]')!)
    expect(within(card()).getByRole('heading', { level: 2 })).toHaveTextContent('Layered Yardangs')
  })

  it('Back and Escape close the card', () => {
    renderAt('/mars?site=site_02')
    expect(card()).toBeInTheDocument()
    fireEvent.click(within(card()).getByRole('button', { name: 'Back' }))
    expect(location()).toBe('/mars')
    noCard()
    fireEvent.click(within(list()).getByRole('button', { name: /Kaiser Crater/ }))
    fireEvent.keyDown(screen.getByRole('region', { name: /map of Mars/ }), { key: 'Escape' })
    expect(location()).toBe('/mars')
  })

  it('keeps the required caveats visible in the footer', () => {
    renderAt('/mars')
    const footer = document.querySelector('.globe-footer')!
    for (const id of ['no_quantum_advantage', 'mars_terrain_proxy']) {
      const c = (caveatsFile as CaveatsFile).caveats.find((x) => x.id === id)!
      expect(footer.querySelector(`[data-caveat="${id}"]`)).toHaveTextContent(c.text)
    }
  })
})

describe('routing between globe and explorer', () => {
  it('"Explore site" opens the explorer on that site', () => {
    renderAt('/mars?site=site_05')
    fireEvent.click(within(card()).getByRole('link', { name: 'Explore site' }))
    expect(location()).toBe('/mars/explore?site=site_05')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Huo Hsing Vallis')
  })

  it('the explorer locator goes back to the globe on the same site', () => {
    renderAt('/mars/explore?site=site_02&model=qcnn')
    const back = screen.getByRole('link', { name: 'Back to globe' })
    expect(back).toHaveAttribute('href', '/mars?site=site_02')
    fireEvent.click(back)
    expect(location()).toBe('/mars?site=site_02')
    expect(within(card()).getByRole('heading', { level: 2 })).toHaveTextContent('Kaiser Crater')
  })

  it.each([
    '/mars?site=site_04&model=qcnn&seed=43',
    '/mars?seed=44',
    '/mars?site=site_01&truth=1',
    '/mars?scope=all',
  ])('old explorer link %s still opens the explorer with its parameters', (path) => {
    renderAt(path)
    expect(location()).toBe(path.replace('/mars?', '/mars/explore?'))
    expect(screen.getByRole('region', { name: /orthoimage strip/ })).toBeInTheDocument()
  })

  it('plain /mars and /mars?site= stay on the globe', () => {
    renderAt('/mars?site=site_01')
    expect(location()).toBe('/mars?site=site_01')
    expect(document.querySelector('.globe-page')).not.toBeNull()
  })

  it('the Moon globe has no location pins', () => {
    renderAt('/moon')
    expect(document.querySelector('.globe-page')).not.toBeNull()
    expect(document.querySelector('.callout, .pin-hit, .footprint, .drape')).toBeNull()
  })
})

describe('the card', () => {
  it('is not rendered until a site is chosen', () => {
    renderAt('/mars')
    noCard()
    fireEvent.click(within(list()).getByRole('button', { name: /Coprates Chasma/ }))
    expect(card()).toHaveTextContent('Coprates Chasma')
  })
})

describe('focus', () => {
  it('a site in the URL on load does not steal focus; choosing one moves focus to the card', () => {
    renderAt('/mars?site=site_02')
    expect(document.activeElement).toBe(document.body)
    fireEvent.click(within(list()).getByRole('button', { name: /Layered Yardangs/ }))
    expect(within(card()).getByRole('heading', { level: 2 })).toHaveFocus()
  })
})
