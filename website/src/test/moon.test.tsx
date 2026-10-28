// Moon explorer (step 9) and Moon globe entry (step 9b). jsdom has no WebGL, so the globe tests
// exercise the flat-map fallback.
import { fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter, useLocation } from 'react-router'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import caveatsFile from '../../public/data/caveats.json'
import summaryFile from '../../public/data/moon/summary.json'
import { App } from '../App'
import { LOCATION_NOTE } from '../components/moon/moonData'
import type { CaveatsFile, MoonClassFile, MoonSummary } from '../data/types'

const PUBLIC = resolve(import.meta.dirname, '../../public')
const summary = summaryFile as unknown as MoonSummary
const caveats = (caveatsFile as CaveatsFile).caveats
const classFile = (c: string): MoonClassFile => JSON.parse(readFileSync(resolve(PUBLIC, `data/moon/classes/${c}.json`), 'utf8'))

const sizes = { clientWidth: 1200, clientHeight: 700 }
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
        return this.classList.contains('globe-stage') || this.classList.contains('mosaic') ? v : 0
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
const mosaic = () => screen.findByRole('group', { name: /test crops$/ })
const results = () => screen.getByRole('complementary', { name: 'Results' })

describe('Moon explorer: class rail, mosaic and controls', () => {
  it('lists the three classes in order with their test-crop counts', () => {
    renderAt('/moon/explore')
    const items = within(screen.getByRole('navigation', { name: 'Classes' })).getAllByRole('link')
    expect(items.map((a) => a.querySelector('.rail-label')!.textContent)).toEqual(['Ejecta', 'Old crater', 'None'])
    expect(items.map((a) => a.querySelector('.rail-detail')!.textContent)).toEqual(['89 test crops', '157 test crops', '533 test crops'])
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Ejecta')
  })

  it('offers the three models in fixed order and ten seeds; choices live in the URL', async () => {
    renderAt('/moon/explore?class=oldcrater')
    const radios = within(screen.getByRole('radiogroup', { name: 'Model' })).getAllByRole('radio')
    expect(radios.map((r) => r.textContent)).toEqual(['CNN', 'Hybrid CNN–QCNN', 'Frozen-embedding MLP'])
    const seed = screen.getByRole('combobox', { name: 'Seed' })
    expect([...seed.querySelectorAll('option')].map((o) => o.value)).toEqual(summary.seeds.map(String))
    fireEvent.click(radios[2])
    fireEvent.change(seed, { target: { value: '47' } })
    fireEvent.click(screen.getByRole('switch', { name: 'Predicted class' }))
    expect(location()).toBe('/moon/explore?class=oldcrater&model=mlp&seed=47&labels=1')
    expect(document.querySelector('[data-winner]')).toBeNull()
  })

  it.each([
    ['ejecta', 'cnn', 'cnn', '42'],
    ['oldcrater', 'qcnn', 'hybrid_qcnn', '45'],
    ['none', 'mlp', 'frozen_embedding_mlp', '51'],
  ] as const)('%s / %s / seed %s: solid outlines exactly where the prediction is right', async (cls, url, key, seed) => {
    renderAt(`/moon/explore?class=${cls}&model=${url}&seed=${seed}`)
    const grid = await mosaic()
    const file = classFile(cls)
    const crops = within(grid).getAllByRole('button')
    expect(crops).toHaveLength(file.crops.length)
    file.crops.forEach((c, i) => {
      const right = c.pred[key][seed].label === cls
      expect(crops[i].hasAttribute('data-correct')).toBe(right)
    })
    // The legend counts are the same predictions.
    const counts = Object.fromEntries(['ejecta', 'oldcrater', 'none'].map((k) => [k, file.crops.filter((c) => c.pred[key][seed].label === k).length]))
    for (const [k, n] of Object.entries(counts)) {
      expect(document.querySelector(`.moon-counts [data-predicted="${k}"] td`)).toHaveTextContent(String(n))
    }
  })

  it('outlines in the active model colour, never a truth colour', async () => {
    renderAt('/moon/explore?model=qcnn')
    expect(await mosaic()).toHaveAttribute('data-model', 'qcnn')
    const css = readFileSync(resolve(import.meta.dirname, '../styles/moon.css'), 'utf8')
    const crop = css.slice(css.indexOf('.crop {'), css.indexOf('}', css.indexOf('.crop {')))
    expect(crop).toMatch(/outline: 2px dashed var\(--swatch\)/)
    expect(css).not.toMatch(/--truth-(safe|unsafe)/)
  })

  it('"Predicted class" labels each crop with its predicted class', async () => {
    renderAt('/moon/explore?class=ejecta&labels=1&seed=43')
    const file = classFile('ejecta')
    const tags = (await mosaic()).querySelectorAll('.crop-tag')
    expect(tags).toHaveLength(file.crops.length)
    expect(tags[0].textContent).toBe(file.crops[0].pred.cnn['43'].label === 'oldcrater' ? 'old crater' : file.crops[0].pred.cnn['43'].label)
  })

  it('the crop inspector lists every model in fixed order; Escape closes it', async () => {
    renderAt('/moon/explore?class=none&seed=44')
    const crops = within(await mosaic()).getAllByRole('button')
    fireEvent.click(crops[3])
    const inspector = screen.getByRole('region', { name: /^Crop / })
    expect(within(inspector).getAllByRole('rowheader').map((r) => r.textContent)).toEqual(['CNN', 'Hybrid', 'MLP'])
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: /^Crop / })).toBeNull()
  })

  it('moves along the mosaic with the arrow keys (one tab stop)', async () => {
    renderAt('/moon/explore?class=oldcrater')
    const grid = await mosaic()
    const crops = within(grid).getAllByRole('button')
    expect(crops.filter((c) => c.tabIndex === 0)).toHaveLength(1)
    crops[0].focus()
    fireEvent.keyDown(crops[0], { key: 'ArrowRight' })
    expect(crops[1]).toHaveFocus()
    fireEvent.keyDown(crops[1], { key: 'End' })
    expect(crops[crops.length - 1]).toHaveFocus()
  })
})

describe('Moon explorer: results panel', () => {
  it('shows balanced accuracy and macro-F1 first, then accuracy', () => {
    renderAt('/moon/explore')
    const heads = within(results()).getAllByRole('heading', { level: 4 }).map((h) => h.textContent)
    expect(heads).toEqual(['Balanced accuracy', 'Macro-F1', 'Accuracy'])
  })

  it.each(['balanced_accuracy', 'macro_f1', 'accuracy'])('%s: the 10-seed mean, SD, range and every seed, per model in fixed order', (key) => {
    renderAt('/moon/explore?seed=46')
    const chart = results().querySelector(`[data-metric="${key}"]`)!
    const rows = [...chart.querySelectorAll('.metric-row')]
    expect(rows.map((r) => r.getAttribute('data-model'))).toEqual(['cnn', 'qcnn', 'mlp'])
    ;(['cnn', 'hybrid_qcnn', 'frozen_embedding_mlp'] as const).forEach((m, i) => {
      expect(Number(rows[i].querySelector('[data-value]')!.getAttribute('data-value'))).toBe(summary.metrics[m][key].mean)
      expect(rows[i].querySelector('.metric-sd')).toHaveTextContent(`± ${summary.metrics[m][key].sd.toFixed(3)}`)
      const r = summary.ranges[m][key as 'accuracy']
      expect(rows[i].querySelector('.metric-range')).toHaveTextContent(`${r.min.toFixed(3)}–${r.max.toFixed(3)}`)
      expect(rows[i].querySelectorAll('.metric-dot')).toHaveLength(10)
      expect(rows[i].querySelectorAll('.metric-dot[data-active]')).toHaveLength(1)
    })
    expect(chart.querySelector('.metric-ref-label')).toHaveTextContent(`Majority baseline: ${summary.metrics.majority_train_baseline[key].mean.toFixed(3)}`)
  })

  it('pins the moon_reproducibility caveat verbatim beside the metrics', () => {
    renderAt('/moon/explore?class=none')
    const c = caveats.find((x) => x.id === 'moon_reproducibility')!
    expect(results().querySelector('.results-caveat[data-caveat="moon_reproducibility"]')).toHaveTextContent(c.text)
  })

  it('renders every Moon and shared caveat verbatim, including the 59,347 parameter count', () => {
    renderAt('/moon/explore')
    const moon = caveats.filter((c) => c.world === 'moon' || c.world === 'both')
    for (const c of moon) expect(document.querySelector(`.caveat-list [data-caveat="${c.id}"]`), c.id).toHaveTextContent(c.text)
    expect(document.querySelector('.caveat-list')).toHaveTextContent('59,347')
  })

  it('marks no winner and shows no paired statistics', () => {
    renderAt('/moon/explore')
    const body = results().querySelector('.results-body')!.textContent!
    expect(body).not.toMatch(/\b(winner|best|wins|outperform|paired|difference)/i)
  })
})

describe('Moon globe entry', () => {
  it('opens /moon on the flat map without WebGL, with no pins', () => {
    renderAt('/moon')
    expect(document.querySelector('.globe-page')).toHaveAttribute('data-mode', 'map')
    expect(screen.getByRole('radio', { name: 'Globe' })).toBeDisabled()
    expect(document.querySelector('.callout, .pin-hit, .footprint, .drape')).toBeNull()
  })

  it('says why nothing is placed on the globe', () => {
    renderAt('/moon')
    expect(screen.getByText(LOCATION_NOTE)).toBeVisible()
  })

  it('shows one card per class, in order, with counts, four sample crops and an explore link', async () => {
    renderAt('/moon')
    const list = screen.getByRole('list', { name: 'Crater classes' })
    const cards = within(list).getAllByRole('listitem')
    expect(cards.map((c) => within(c).getByRole('heading', { level: 2 }).textContent)).toEqual(['Ejecta', 'Old crater', 'None'])
    const want = { ejecta: 89, oldcrater: 157, none: 533 }
    for (const [i, cls] of (['ejecta', 'oldcrater', 'none'] as const).entries()) {
      expect(cards[i]).toHaveTextContent(`${want[cls]} test crops`)
      expect(within(cards[i]).getByRole('link', { name: 'Explore class' })).toHaveAttribute('href', `/moon/explore?class=${cls}`)
      const crops = await within(cards[i]).findAllByRole('img')
      const first = classFile(cls).crops.slice(0, 4).map((c) => c.sample_id.split('/').pop())
      expect(crops.map((c) => c.getAttribute('aria-label')!.split(' ').pop())).toEqual(first)
    }
  })

  it('keeps the required Moon caveats in the footer, verbatim', () => {
    renderAt('/moon')
    const footer = document.querySelector('.globe-footer')!
    for (const id of ['moon_reproducibility', 'moon_provenance', 'no_quantum_advantage']) {
      expect(footer.querySelector(`[data-caveat="${id}"]`)).toHaveTextContent(caveats.find((c) => c.id === id)!.text)
    }
  })

  it('"Explore class" opens the explorer on that class; "Back to the Moon" returns', () => {
    renderAt('/moon')
    const card = screen.getAllByRole('listitem').find((li) => li.getAttribute('data-class') === 'none')!
    fireEvent.click(within(card).getByRole('link', { name: 'Explore class' }))
    expect(location()).toBe('/moon/explore?class=none')
    fireEvent.click(screen.getByRole('link', { name: 'Back to the Moon' }))
    expect(location()).toBe('/moon?class=none')
    expect(document.querySelector('.class-card[data-class="none"]')).toHaveAttribute('data-current')
  })

  it.each(['/moon?class=none&model=mlp', '/moon?seed=44', '/moon?labels=1&class=ejecta'])(
    'old explorer link %s still opens the explorer with its parameters',
    (path) => {
      renderAt(path)
      expect(location()).toBe(path.replace('/moon?', '/moon/explore?'))
    },
  )
})
