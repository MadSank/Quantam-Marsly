// Enabled from build step 10 (PLAN.md "Tests"): every data source carries a licence and a
// citation, and nothing is left to confirm.
import { describe, expect, it } from 'vitest'
import type { AboutFile } from '../data/types'
import { loadData } from './data'

const about = loadData<AboutFile>('about.json')

describe('about.json sources', () => {
  it('lists every data source the site uses', () => {
    const names = about.sources.map((s) => s.name).join(' | ')
    for (const needle of ['HiRISE', 'LROCNet', 'Viking MDIM', 'MOLA', 'CGI Moon Kit', 'LOLA']) expect(names).toContain(needle)
  })

  it.each(about.sources.map((s) => [s.name, s] as const))('%s: licence, citation, credit and links are filled in', (_, s) => {
    for (const field of ['provider', 'licence', 'credit', 'citation', 'url', 'licence_url', 'used_for'] as const) {
      expect(s[field], field).toBeTruthy()
      expect(s[field], field).not.toMatch(/TO CONFIRM/i)
    }
    expect(s.url).toMatch(/^https:\/\//)
    expect(s.licence_url).toMatch(/^https:\/\//)
  })

  it('leaves nothing marked TO CONFIRM anywhere in about.json', () => {
    expect(JSON.stringify(about)).not.toMatch(/TO CONFIRM/i)
  })

  it('states the LROCNet licence and records that the crops were resized (CC BY asks for changes to be noted)', () => {
    const lroc = about.sources.find((s) => s.name.includes('LROCNet'))!
    expect(lroc.licence).toMatch(/CC BY 4\.0/)
    expect(lroc.citation).toMatch(/10\.5281\/zenodo\.7041842/)
    expect(lroc.used_for).toMatch(/resized/)
  })

  it('cites the relief sources by their data set IDs', () => {
    expect(about.sources.find((s) => s.name.includes('MOLA'))!.citation).toMatch(/MGS-M-MOLA-5-MEGDR-L3-V1\.0/)
    expect(about.sources.find((s) => s.name.includes('LOLA'))!.citation).toMatch(/LRO-L-LOLA-4-GDR-V1\.0/)
  })
})
