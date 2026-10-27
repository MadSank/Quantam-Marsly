// The paired-site statistical section removed from the Streamlit app must not return here,
// in code, in data, or as prose (PLAN.md "Tests").
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { WEBSITE_ROOT, walk } from './data'

const TEXT_FILE = /\.(ts|tsx|js|jsx|css|html|json|md|svg|txt)$/
// Test files under src/test/ name the banned patterns, so they are excluded from the scan.
const SCANNED = [...walk(join(WEBSITE_ROOT, 'src')), ...walk(join(WEBSITE_ROOT, 'public'))].filter(
  (f) => TEXT_FILE.test(f) && !f.startsWith('src/test/'),
)

const BANNED: Array<[string, RegExp]> = [
  ['paired_differences', /paired_differences/i],
  ['Paired Site Differences', /paired site differences/i],
  ['t95 / t95_low / t95_high', /\bt95(_low|_high)?\b/i],
  ['t interval / t-interval', /\bt[ -]intervals?\b/i],
  ['cross zero', /\bcross(es|ing)? zero\b/i],
]

describe('removed statistical section stays removed', () => {
  it('scans a non-trivial set of files', () => {
    expect(SCANNED.length).toBeGreaterThan(10)
  })

  it.each(BANNED)('no file contains %s', (_, pattern) => {
    const hits = SCANNED.filter((f) => pattern.test(readFileSync(join(WEBSITE_ROOT, f), 'utf8')))
    expect(hits).toEqual([])
  })
})
