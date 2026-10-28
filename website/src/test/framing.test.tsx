// Neutral presentation (PLAN.md "Rules for presenting the science"): both models side by side in
// fixed order whatever their values, no winner marks, a direction on every metric, recall /
// FPR / coverage together, both regimes, and the mixed-results caveat always on screen.
import { fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter, useLocation } from 'react-router'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import caveatsFile from '../../public/data/caveats.json'
import headlineFile from '../../public/data/mars/headline.json'
import sitesFile from '../../public/data/mars/sites.json'
import { App } from '../App'
import type { CaveatsFile, MarsHeadline, MarsSiteFile, MarsSitesFile } from '../data/types'

const PUBLIC = resolve(import.meta.dirname, '../../public')
const sites = (sitesFile as MarsSitesFile).sites
const headline = headlineFile as unknown as MarsHeadline
const mixed = (caveatsFile as CaveatsFile).caveats.find((c) => c.id === 'mars_mixed_results')!

beforeAll(() => {
  vi.stubGlobal('fetch', async (url: string) => {
    const path = resolve(PUBLIC, decodeURI(String(url)).replace(/^\//, ''))
    return { ok: true, status: 200, json: async () => JSON.parse(readFileSync(path, 'utf8')) }
  })
})
afterAll(() => vi.unstubAllGlobals())

function Location() {
  return <output data-testid="location">{useLocation().search}</output>
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <Location />
    </MemoryRouter>,
  )
}

const panel = () => screen.getByRole('complementary', { name: 'Results' })
const loaded = () => within(panel()).findAllByText('Mean slope RMSE')
const siteFile = (id: string): MarsSiteFile => JSON.parse(readFileSync(resolve(PUBLIC, `data/mars/sites/${id}.json`), 'utf8'))

describe('framing', () => {
  it.each(sites.map((s) => s.site_id))('%s: every metric lists CNN then the hybrid, whatever the values', async (id) => {
    renderAt(`/mars?site=${id}`)
    await loaded()
    const metrics = panel().querySelectorAll('.metric')
    expect(metrics.length).toBe(3 + 2 * 3)
    for (const m of metrics) {
      expect([...m.querySelectorAll('.metric-row')].map((r) => r.getAttribute('data-model'))).toEqual(['cnn', 'qcnn'])
    }
  })

  it('the data has metrics where each model is larger, so fixed order is not a sort', () => {
    const keys = ['RMSE_mean_slope_5m', 'RMSE_roughness_m', 'RMSE_hazard_fraction', 'safe_recall', 'false_positive_rate', 'predicted_safe_fraction']
    const larger = new Set(
      sites.flatMap((s) => {
        const f = siteFile(s.site_id).mean.fixed
        return keys.map((k) => (f.CNN[k]! > f.CNN_QCNN[k]! ? 'cnn' : 'qcnn'))
      }),
    )
    expect(larger).toEqual(new Set(['cnn', 'qcnn']))
  })

  it('marks no winner anywhere', async () => {
    renderAt('/mars?site=site_02')
    await loaded()
    expect(document.querySelector('[data-winner], .winner, .best')).toBeNull()
    // The charts and their labels only; caveats quote the audit ("no clear overall winner").
    expect(panel().querySelector('.results-body')!.textContent).not.toMatch(/\b(winner|best|wins|outperform)/i)
  })

  it('gives every metric a direction, or says it has none', async () => {
    renderAt('/mars')
    await loaded()
    for (const m of panel().querySelectorAll('.metric')) {
      expect(m.querySelector('.metric-direction')!.textContent).toMatch(/^(lower is better|higher is better|read with recall and false-positive rate)$/)
    }
  })

  it('shows recall, false-positive rate and coverage together, under both regimes', async () => {
    renderAt('/mars')
    await loaded()
    for (const name of [/fixed threshold/i, /validation-calibrated/i]) {
      const group = within(panel()).getByRole('region', { name })
      const labels = within(group).getAllByRole('heading', { level: 4 }).map((h) => h.textContent)
      expect(labels).toEqual(['Safe recall', 'False-positive rate', 'Coverage, share predicted safe'])
    }
  })

  it('keeps the mixed-results caveat, verbatim, on screen', async () => {
    renderAt('/mars')
    expect(panel().querySelector('[data-caveat="mars_mixed_results"]')).toHaveTextContent(mixed.text)
    fireEvent.click(within(panel()).getByRole('radio', { name: 'All five sites' }))
    expect(panel().querySelector('[data-caveat="mars_mixed_results"]')).toHaveTextContent(mixed.text)
  })

  it('site scope: bars show the seed mean from the site file', async () => {
    renderAt('/mars?site=site_03')
    await loaded()
    const f = siteFile('site_03')
    const slope = panel().querySelector('[data-metric="RMSE_mean_slope_5m"]')!
    const values = [...slope.querySelectorAll('.metric-value [data-value]')].map((e) => Number(e.getAttribute('data-value')))
    expect(values).toEqual([f.mean.fixed.CNN.RMSE_mean_slope_5m, f.mean.fixed.CNN_QCNN.RMSE_mean_slope_5m])
    expect(slope.querySelectorAll('.metric-dot')).toHaveLength(6)
  })

  it('five-site scope: bars show the audited headline mean ± SD, and the scope is in the URL', async () => {
    renderAt('/mars?site=site_01')
    await loaded()
    fireEvent.click(within(panel()).getByRole('radio', { name: 'All five sites' }))
    expect(screen.getByTestId('location').textContent).toContain('scope=all')
    await within(panel()).findByText(/All five test sites/)
    const recall = await screen.findByText((_, el) => el?.getAttribute('data-metric') === 'safe_recall' && el.closest('[aria-labelledby]')?.textContent?.includes('fixed threshold') === true && el.querySelectorAll('.metric-dot').length === 10)
    const values = [...recall.querySelectorAll('.metric-value [data-value]')].map((e) => Number(e.getAttribute('data-value')))
    expect(values).toEqual([headline.fixed.CNN.safe_recall.mean, headline.fixed.CNN_QCNN.safe_recall.mean])
    expect(recall.querySelectorAll('.metric-sd')).toHaveLength(2)
  })
})

describe('Mars caveats in the explorer', () => {
  it('renders every Mars and shared caveat, verbatim', () => {
    renderAt('/mars')
    const all = (caveatsFile as CaveatsFile).caveats.filter((c) => c.world === 'mars' || c.world === 'both')
    expect(all.length).toBeGreaterThan(5)
    for (const c of all) {
      const el = document.querySelector(`.caveat-list [data-caveat="${c.id}"]`)
      expect(el, c.id).not.toBeNull()
      expect(el).toHaveTextContent(c.text)
    }
  })
})
