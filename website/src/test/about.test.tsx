import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import aboutFile from '../../public/data/about.json'
import caveatsFile from '../../public/data/caveats.json'
import summaryFile from '../../public/data/moon/summary.json'
import { App } from '../App'
import type { AboutFile, CaveatsFile, MoonSummary } from '../data/types'

const about = aboutFile as unknown as AboutFile
const caveats = (caveatsFile as CaveatsFile).caveats
const summary = summaryFile as unknown as MoonSummary

function renderAbout() {
  return render(
    <MemoryRouter initialEntries={['/about']}>
      <App />
    </MemoryRouter>,
  )
}

describe('About page', () => {
  it('stays on the neutral shell', () => {
    const { container } = renderAbout()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('About the study')
    expect(container.querySelector('[data-world]')).toBeNull()
  })

  it('renders every caveat verbatim with its source file', () => {
    renderAbout()
    for (const c of caveats) {
      const el = document.querySelector(`#caveats [data-caveat="${c.id}"]`)
      expect(el, c.id).not.toBeNull()
      expect(el).toHaveTextContent(c.text)
      expect(el).toHaveTextContent(c.source)
    }
    expect(screen.getByRole('region', { name: 'Caveats' })).toHaveTextContent('59,347')
  })

  it('renders every method block from about.json', () => {
    renderAbout()
    const method = screen.getByRole('region', { name: 'Method' })
    for (const m of about.method) {
      expect(within(method).getByRole('heading', { name: m.title })).toBeInTheDocument()
      for (const p of m.paragraphs) expect(method).toHaveTextContent(p)
    }
  })

  it('makes no performance claims in the method text', () => {
    const text = about.method.flatMap((m) => m.paragraphs).join(' ')
    expect(text).not.toMatch(/\b(better|outperform\w*|superior|wins?|winner|beats|faster|speed-?up|improves|more accurate)\b/i)
    expect(text).toMatch(/no quantum advantage is claimed/)
  })

  it('shows timings only as separately measured and not comparable, in fixed model order', () => {
    renderAbout()
    const timings = screen.getByRole('region', { name: 'Timings' })
    expect(timings).toHaveTextContent('Separately measured, not comparable.')
    expect(timings).toHaveTextContent(caveats.find((c) => c.id === 'moon_timing')!.text)
    const rows = within(timings).getAllByRole('rowheader').map((r) => r.textContent)
    expect(rows).toEqual(['CNN', 'Hybrid CNN–QCNN', 'Frozen-embedding MLP'])
    const cnn = Object.values(summary.timing.milliseconds_per_image.cnn)
    const mean = cnn.reduce((a, b) => a + b, 0) / cnn.length
    expect(within(timings).getAllByRole('row')[1]).toHaveTextContent(mean.toFixed(3))
  })

  it('lists every data source with licence, citation and links', () => {
    renderAbout()
    const sources = screen.getByRole('region', { name: 'Data sources' })
    const items = within(sources).getAllByRole('listitem')
    expect(items).toHaveLength(about.sources.length)
    about.sources.forEach((s, i) => {
      expect(items[i]).toHaveTextContent(s.name)
      expect(items[i]).toHaveTextContent(s.licence)
      expect(items[i]).toHaveTextContent(s.citation)
      expect(within(items[i]).getByRole('link', { name: 'Terms' })).toHaveAttribute('href', s.licence_url)
    })
  })

  it('the section index moves focus to the chosen section', () => {
    renderAbout()
    const toc = screen.getByRole('navigation', { name: 'On this page' })
    Element.prototype.scrollIntoView = () => {}
    fireEvent.click(within(toc).getByRole('link', { name: 'Data sources' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Data sources' })).toHaveFocus()
  })
})
