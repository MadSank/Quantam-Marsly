// Required caveats exist with the approved wording. Rendering checks (each caveat shown on
// About and in its world) are added when those pages are built (build steps 8–10).
import { describe, expect, it } from 'vitest'
import type { CaveatsFile } from '../data/types'
import { loadData } from './data'

const { caveats } = loadData<CaveatsFile>('caveats.json')
const byId = new Map(caveats.map((c) => [c.id, c]))

const REQUIRED: Array<[string, 'mars' | 'moon' | 'both']> = [
  ['safe_proxy', 'mars'],
  ['mars_mixed_results', 'mars'],
  ['mars_pipeline_comparison', 'mars'],
  ['mars_no_matched_control', 'mars'],
  ['mars_sites', 'mars'],
  ['mars_terrain_proxy', 'mars'],
  ['no_quantum_advantage', 'both'],
  ['moon_param_count', 'moon'],
  ['moon_timing', 'moon'],
  ['moon_imbalance', 'moon'],
  ['moon_training_regime', 'moon'],
  ['moon_mixed_results', 'moon'],
  ['moon_seeds', 'moon'],
  ['moon_provenance', 'moon'],
  ['moon_test_embeddings', 'moon'],
  ['moon_reproducibility', 'moon'],
]

describe('caveats.json', () => {
  it.each(REQUIRED)('%s exists for %s', (id, world) => {
    const c = byId.get(id)
    expect(c, id).toBeDefined()
    expect(c!.world).toBe(world)
    expect(c!.text.length).toBeGreaterThan(20)
    expect(c!.source).toMatch(/^(FINAL_READINESS_AUDIT\.md|visualization_app\/README\.md|experiments\/moon\/README\.md)$/)
  })

  it('uses the approved mixed-results wording (audit §1, no interval language)', () => {
    expect(byId.get('mars_mixed_results')!.text).toBe(
      'Numerically, the QCNN reduces the reported regression errors and false-positive rate, ' +
        'but the CNN has higher safe recall and coverage. ' +
        'These results therefore do not identify a clear overall winner.',
    )
    expect(byId.get('moon_mixed_results')!.text).toBe(
      'Ten seeds on one supplied split are descriptive evidence, not proof of a general advantage.',
    )
  })

  it('states the corrected Moon CNN parameter count', () => {
    expect(byId.get('moon_param_count')!.text).toContain('59,347')
  })

  it('labels "safe" as a proxy and disclaims quantum advantage', () => {
    expect(byId.get('safe_proxy')!.text).toMatch(/derived proxy label/)
    expect(byId.get('no_quantum_advantage')!.text).toMatch(/No quantum advantage or superiority is claimed/)
    expect(byId.get('moon_timing')!.text).toMatch(/not an apples-to-apples/)
  })

  it('quotes the audit addendum: 10 deterministic seeds, large spread, no model clearly ahead', () => {
    const c = byId.get('moon_reproducibility')!
    expect(c.source).toBe('FINAL_READINESS_AUDIT.md')
    expect(c.text).toMatch(/come from 10 deterministic seeds/)
    expect(c.text).toMatch(/spread across seeds is large/)
    expect(c.text).toMatch(/no model is clearly ahead/)
    expect(c.text).toMatch(/do not indicate any quantum contribution/)
  })

  it('no Moon caveat still describes the superseded three-seed run', () => {
    for (const c of caveats.filter((c) => c.world === 'moon')) expect(c.text).not.toMatch(/three seeds|three-seed/i)
  })

  it.todo('renders every caveat on About (step 10)')
})
