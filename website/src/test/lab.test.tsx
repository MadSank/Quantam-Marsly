// Lab page (step 11; PLAN.md "lab.test.tsx"): parameter counts, permalinks that resolve to the
// excerpted lines, no performance claims, no timing fields, and the Moon code-history note.
import { fireEvent, render, screen, within } from '@testing-library/react'
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import type { LabCurves, LabExcerpts, LabParams } from '../data/types'
import { DATA_ROOT, loadData, walk, WEBSITE_ROOT } from './data'

const PUBLIC = resolve(import.meta.dirname, '../../public')
const REPO_ROOT = resolve(WEBSITE_ROOT, '..')
const params = loadData<LabParams>('lab/params.json')
const curves = loadData<LabCurves>('lab/curves.json')
const excerpts = loadData<LabExcerpts>('lab/excerpts.json')

const AUDIT = { marsCnn: 11_242_563, marsHybrid: 4_152, moonCnn: 59_347, moonFrozen: 195, moonHybrid: 568, moonMlp: 547 }

// PLAN.md "No performance claims". The two direction labels are removed before the scan.
const CLAIMS = /\b(better|outperform\w*|superior|wins|winner|beats|faster|speed-?up|advantage|improves|more accurate)\b/i
const ALLOWED = /\b(lower|higher) is better\b/g
// The footer's caveat line names quantum advantage only to deny it.
const FOOTER_CAVEAT = 'No quantum advantage claimed; circuits simulated.'

const git = (...args: string[]) => execFileSync('git', args, { cwd: REPO_ROOT, encoding: 'utf8' })

beforeAll(() => {
  vi.stubGlobal('fetch', async (url: string) => {
    const path = resolve(PUBLIC, decodeURI(String(url)).replace(/^\//, ''))
    const body = readFileSync(path, 'utf8')
    return { ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body }
  })
})
afterAll(() => vi.unstubAllGlobals())

async function renderLab() {
  const view = render(
    <MemoryRouter initialEntries={['/lab']}>
      <App />
    </MemoryRouter>,
  )
  await screen.findByRole('heading', { level: 2, name: 'Code' }, { timeout: 5000 })
  return view
}

describe('Lab parameter counts', () => {
  it('params.json holds the audit constants', () => {
    expect(params.mars.cnn.trainable).toBe(AUDIT.marsCnn)
    expect(params.mars.hybrid.trainable).toBe(AUDIT.marsHybrid)
    expect(params.mars.hybrid.bottleneck + params.mars.hybrid.qcnn_head).toBe(AUDIT.marsHybrid)
    expect(params.moon.cnn.trainable_during_training).toBe(AUDIT.moonCnn)
    expect(params.moon.cnn.trainable_post_freeze).toBe(AUDIT.moonFrozen)
    expect(params.moon.hybrid_head).toBe(AUDIT.moonHybrid)
    expect(params.moon.mlp_head).toBe(AUDIT.moonMlp)
  })

  it('the page shows the same counts', async () => {
    await renderLab()
    const table = screen.getByRole('table', { name: /Parameters counted/ })
    const row = (id: string) => table.querySelector(`[data-param="${id}"]`)!
    expect(row('mars-cnn')).toHaveTextContent('11,242,563')
    expect(row('mars-hybrid')).toHaveTextContent('4,152')
    expect(row('moon-cnn')).toHaveTextContent('59,347')
    expect(row('moon-cnn')).toHaveTextContent('195')
    expect(row('moon-hybrid')).toHaveTextContent('568')
    expect(row('moon-mlp')).toHaveTextContent('547')
  })
})

describe('Lab permalinks', () => {
  const commit = excerpts.commit
  it('the commit exists in the repository', () => {
    expect(commit).toMatch(/^[0-9a-f]{40}$/)
    git('cat-file', '-e', `${commit}^{commit}`)
  })

  it.each(excerpts.excerpts.map((e) => [e.id, e] as const))('%s points at the excerpted lines', (_, e) => {
    expect(e.permalink).toBe(`https://github.com/MadSank/Quantam-Marsly/blob/${commit}/${e.path}#L${e.start}-L${e.end}`)
    git('cat-file', '-e', `${commit}:${e.path}`)
    const lines = git('show', `${commit}:${e.path}`).split('\n')
    if (lines.at(-1) === '') lines.pop()
    expect(e.start).toBeGreaterThanOrEqual(1)
    expect(e.start).toBeLessThanOrEqual(e.end)
    expect(e.end).toBeLessThanOrEqual(lines.length)
    expect(lines.slice(e.start - 1, e.end).join('\n')).toBe(e.code.replaceAll('\r', ''))
  })

  it('every excerpt is highlighted at build time and rendered with its link', async () => {
    await renderLab()
    for (const e of excerpts.excerpts) {
      expect(e.html, e.id).toMatch(/^<pre class="shiki/)
      const article = document.querySelector(`[data-excerpt="${e.id}"]`) as HTMLElement
      expect(within(article).getByRole('link', { name: 'Open these lines on GitHub' })).toHaveAttribute('href', e.permalink)
    }
  })

  it('every pipeline stage links to code', async () => {
    await renderLab()
    for (const stage of document.querySelectorAll('.pipeline-stage')) {
      const links = within(stage as HTMLElement).getAllByRole('link')
      expect(links.length).toBeGreaterThan(0)
      for (const a of links) expect(excerpts.excerpts.map((e) => e.permalink)).toContain(a.getAttribute('href'))
    }
  })
})

describe('Lab makes no performance claims', () => {
  it('in the rendered page', async () => {
    const { container } = await renderLab()
    const text = (container.textContent ?? '').replace(FOOTER_CAVEAT, '').replace(ALLOWED, '')
    expect(text).not.toMatch(CLAIMS)
  })

  it('in the lab data notes', () => {
    const files = walk(`${DATA_ROOT}/lab`).filter((f) => f.endsWith('.json'))
    expect(files.length).toBeGreaterThan(5)
    for (const f of files) {
      const raw = readFileSync(resolve(WEBSITE_ROOT, f), 'utf8')
      expect(raw, f).not.toMatch(CLAIMS)
    }
  })

  it('shows only the two direction labels, on every curve', async () => {
    await renderLab()
    const directions = [...document.querySelectorAll('.curve-direction')].map((d) => d.textContent)
    expect(directions.length).toBe(4)
    for (const d of directions) expect(['lower is better', 'higher is better']).toContain(d)
  })
})

describe('Lab training curves', () => {
  it('carry no wall-time or learning-rate fields', () => {
    const keys = new Set<string>()
    for (const world of [curves.mars, curves.moon])
      for (const run of Object.values(world)) for (const c of Object.values(run) as object[]) Object.keys(c).forEach((k) => keys.add(k))
    expect([...keys].sort()).toEqual(['epoch', 'train_loss', 'val_loss', 'val_macro_f1'])
    expect(JSON.stringify(curves)).not.toMatch(/time|seconds|_lr|"lr"|learning_rate/i)
  })

  it('cover every Mars run and every Moon seed', () => {
    expect(Object.keys(curves.mars)).toHaveLength(15)
    expect(Object.keys(curves.moon)).toEqual(['42', '43', '44', '45', '46', '47', '48', '49', '50', '51'])
  })

  it('switch with the fold selector, models in fixed order', async () => {
    await renderLab()
    const group = screen.getByRole('group', { name: 'Mars fold' })
    fireEvent.click(within(group).getByRole('button', { name: /Fold 3/ }))
    expect(within(group).getByRole('button', { name: /Fold 3/ })).toHaveAttribute('aria-pressed', 'true')
    const marsBlock = group.closest('.curve-block') as HTMLElement
    const series = [...marsBlock.querySelectorAll('.curve')][0].querySelectorAll('.curve-series')
    expect([...series].map((s) => s.getAttribute('data-model'))).toEqual(['cnn', 'qcnn'])
    const points = series[0].querySelectorAll('circle').length
    expect(points).toBe(curves.mars['3_42'].cnn.epoch.length)
  })
})

describe('Lab Moon code history', () => {
  const readme = git('show', `${excerpts.commit}:experiments/moon/README.md`).replaceAll('\r', '').split('\n')
  const start = readme.findIndex((l) => l.trim() === '## Code history')
  let end = readme.findIndex((l, i) => i > start && l.startsWith('## '))
  if (end < 0) end = readme.length
  const section = readme.slice(start, end).join('\n').trimEnd()

  it('every Moon excerpt carries the README section verbatim', () => {
    const moon = excerpts.excerpts.filter((e) => e.path.startsWith('experiments/moon/'))
    expect(moon.length).toBeGreaterThan(0)
    for (const e of moon) {
      expect(e.history_note?.text, e.id).toBe(section)
      expect(e.history_note?.permalink).toMatch(new RegExp(`/blob/${excerpts.commit}/experiments/moon/README\\.md#L${start + 1}-L\\d+$`))
    }
  })

  it('the page shows it next to each Moon excerpt', async () => {
    await renderLab()
    for (const e of excerpts.excerpts) {
      const article = document.querySelector(`[data-excerpt="${e.id}"]`) as HTMLElement
      const note = article.querySelector('[data-history-note]')
      if (e.path.startsWith('experiments/moon/')) {
        expect(note, e.id).not.toBeNull()
        expect(note).toHaveTextContent('What changed after the reported run.')
        expect(within(note as HTMLElement).getByRole('link')).toHaveAttribute('href', e.history_note!.permalink)
      } else expect(note, e.id).toBeNull()
    }
  })
})

describe('Lab shell', () => {
  it('is the neutral shell with the circuit drawing inlined', async () => {
    const { container } = await renderLab()
    expect(container.querySelector('[data-world]')).toBeNull()
    expect(await screen.findByRole('img', { name: /QCNN circuit on 8 qubits/ })).toContainHTML('<svg')
  })
})
