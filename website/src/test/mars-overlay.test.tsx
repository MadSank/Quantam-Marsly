import { fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter, useLocation } from 'react-router'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import caveatsFile from '../../public/data/caveats.json'
import { App } from '../App'
import type { CaveatsFile, MarsSiteFile } from '../data/types'

const PUBLIC = resolve(import.meta.dirname, '../../public')
const safeProxy = (caveatsFile as CaveatsFile).caveats.find((c) => c.id === 'safe_proxy')!

// Serve public/data from disk, and give the pan/zoom viewport a size (jsdom has no layout).
const sizes = { clientWidth: 900, clientHeight: 700 }
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
        return this.classList.contains('panzoom-viewport') ? v : 0
      },
    })
})
afterAll(() => {
  vi.unstubAllGlobals()
  for (const [k, d] of Object.entries(originals)) Object.defineProperty(Element.prototype, k, d)
})

function Location() {
  const loc = useLocation()
  return <output data-testid="location">{loc.search}</output>
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <Location />
    </MemoryRouter>,
  )
}

const search = () => Object.fromEntries(new URLSearchParams(screen.getByTestId('location').textContent!))
const legend = () => screen.getByRole('region', { name: 'Overlay' })
const viewer = () => screen.getByRole('region', { name: /orthoimage strip/ })
const siteFile = (id: string): MarsSiteFile => JSON.parse(readFileSync(resolve(PUBLIC, `data/mars/sites/${id}.json`), 'utf8'))

describe('Mars overlay controls', () => {
  it('offers both models in fixed order, CNN selected by default', () => {
    renderAt('/mars')
    const radios = within(screen.getByRole('radiogroup', { name: 'Model' })).getAllByRole('radio')
    expect(radios.map((r) => r.textContent)).toEqual(['CNN', 'Hybrid CNN–QCNN'])
    expect(radios[0]).toHaveAttribute('aria-checked', 'true')
    expect(document.querySelector('[data-winner]')).toBeNull()
  })

  it('keeps model, seed and truth in the URL', async () => {
    renderAt('/mars?site=site_02')
    await screen.findByText('Predicted safe, proxy safe')
    fireEvent.click(screen.getByRole('radio', { name: 'Hybrid CNN–QCNN' }))
    fireEvent.change(await screen.findByRole('combobox', { name: 'Seed' }), { target: { value: '43' } })
    fireEvent.click(screen.getByRole('switch', { name: 'Ground truth' }))
    expect(search()).toEqual({ site: 'site_02', model: 'qcnn', seed: '43', truth: '1' })
    fireEvent.click(screen.getByRole('switch', { name: 'Ground truth' }))
    expect(search()).toEqual({ site: 'site_02', model: 'qcnn', seed: '43' })
  })

  it('reads model and seed from the URL', async () => {
    renderAt('/mars?site=site_03&model=qcnn&seed=44')
    expect(screen.getByRole('radio', { name: 'Hybrid CNN–QCNN' })).toHaveAttribute('aria-checked', 'true')
    expect(await within(legend()).findByText(/seed/)).toHaveTextContent('Hybrid CNN–QCNN, seed 44. Fixed-threshold decisions.')
  })

  it('leaves the ground-truth layer off by default', () => {
    renderAt('/mars')
    expect(screen.getByRole('switch', { name: 'Ground truth' })).not.toBeChecked()
    expect(within(legend()).queryByText(/Ground truth, corner mark/)).toBeNull()
  })
})

describe('Mars overlay legend', () => {
  it.each([
    ['site_01', 'cnn', 'CNN', '42'],
    ['site_05', 'qcnn', 'CNN_QCNN', '44'],
  ])('%s %s seed %s: counts are the fixed-threshold TP/FP/FN/TN', async (site, key, model, seed) => {
    renderAt(`/mars?site=${site}&model=${key}&seed=${seed}`)
    const row = siteFile(site).per_seed.find((r) => r.regime === 'fixed' && r.model === model && r.seed === Number(seed))!
    await within(legend()).findByText((_, el) => el?.getAttribute('data-outcome') === 'safe-agree' && /\d/.test(el.textContent ?? ''))
    const count = (o: string) => Number(legend().querySelector(`[data-outcome="${o}"] .legend-count`)!.textContent!.replace(/,/g, ''))
    expect([count('safe-agree'), count('safe-wrong'), count('unsafe-wrong'), count('unsafe-agree')]).toEqual([row.TP, row.FP, row.FN, row.TN])
  })

  it('explains every encoding and shows truth rows only with the truth layer', async () => {
    renderAt('/mars?truth=1')
    for (const label of ['Predicted safe, proxy safe', 'Predicted safe, proxy unsafe', 'Predicted unsafe, proxy safe', 'Predicted unsafe, proxy unsafe'])
      expect(within(legend()).getByText(label)).toBeInTheDocument()
    expect(within(legend()).getByText('proxy safe')).toBeInTheDocument()
    expect(within(legend()).getByText('proxy unsafe')).toBeInTheDocument()
  })

  it('always shows the safe-proxy caveat verbatim, outside the drawer', () => {
    renderAt('/mars')
    const caveat = legend().querySelector('[data-caveat="safe_proxy"]')!
    expect(caveat).toHaveTextContent(safeProxy.text)
    expect(caveat.closest('[role="dialog"]')).toBeNull()
  })
})

describe('Mars tile inspector', () => {
  it('Enter inspects the tile at the centre; Close and Escape dismiss it', async () => {
    renderAt('/mars?site=site_05&seed=43')
    await within(legend()).findByText((_, el) => el?.getAttribute('data-outcome') === 'safe-agree' && /\d/.test(el.textContent ?? ''))
    fireEvent.keyDown(viewer(), { key: 'Enter' })
    const inspector = await screen.findByRole('region', { name: /^Tile site_05_r\d+_c\d+$/ })
    const heads = within(inspector).getAllByRole('columnheader').map((h) => h.textContent)
    expect(heads).toEqual(['Quantity', 'Target', 'CNN', 'Hybrid CNN–QCNN'])
    expect(within(inspector).getByRole('rowheader', { name: 'Safe (proxy)' })).toBeInTheDocument()
    fireEvent.click(within(inspector).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('region', { name: /^Tile / })).toBeNull()

    fireEvent.keyDown(viewer(), { key: 'Enter' })
    expect(await screen.findByRole('region', { name: /^Tile / })).toBeInTheDocument()
    fireEvent.keyDown(document.body, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: /^Tile / })).toBeNull()
  })
})
