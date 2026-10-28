import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { countOutcomes } from '../components/overlay/draw'
import { OUTCOME_ROWS, outcome, predictionStyle, truthStyle, type Outcome } from '../components/overlay/style'
import type { MarsSiteFile, MarsTilesFile } from '../data/types'

const ALL: Outcome[] = ['safe-agree', 'safe-wrong', 'unsafe-wrong', 'unsafe-agree']
const pattern = (o: Outcome) => {
  const s = predictionStyle(o)
  return `${s.fill}/${s.hatch}/${s.outline}`
}

describe('overlay encodings', () => {
  it('maps prediction and proxy label to the four outcomes', () => {
    expect(outcome(1, 1)).toBe('safe-agree')
    expect(outcome(1, 0)).toBe('safe-wrong')
    expect(outcome(0, 1)).toBe('unsafe-wrong')
    expect(outcome(0, 0)).toBe('unsafe-agree')
  })

  it('never gives a prediction layer a truth colour', () => {
    for (const o of ALL) expect(['model', null]).toContain(predictionStyle(o).colour)
  })

  it('draws disagreements with a pattern, not only a different hue', () => {
    // Every outcome has its own fill/hatch/outline pattern; colour plays no part in telling them apart.
    expect(new Set(ALL.map(pattern)).size).toBe(4)
    expect(predictionStyle('safe-wrong').hatch).toBe(true)
    expect(predictionStyle('safe-agree').hatch).toBe(false)
    expect(predictionStyle('unsafe-wrong').outline).toBe('dashed')
    expect(predictionStyle('safe-agree').outline).toBe('solid')
    expect(predictionStyle('safe-wrong').colour).toBe(predictionStyle('safe-agree').colour)
    expect(predictionStyle('unsafe-wrong').colour).toBe(predictionStyle('safe-agree').colour)
  })

  it('leaves predicted-unsafe agreements unfilled', () => {
    expect(predictionStyle('unsafe-agree')).toMatchObject({ fill: false, hatch: false, outline: 'none' })
    expect(predictionStyle('unsafe-wrong').fill).toBe(false)
  })

  it('keeps truth colours on the truth layer', () => {
    expect(truthStyle(1).colour).toBe('truth-safe')
    expect(truthStyle(0).colour).toBe('truth-unsafe')
  })

  it('lists every outcome once in the legend', () => {
    expect(OUTCOME_ROWS.map((r) => r.outcome)).toEqual(ALL)
  })
})

describe('overlay data', () => {
  const dir = resolve(import.meta.dirname, '../../public/data/mars')
  const ids = readdirSync(resolve(dir, 'tiles')).map((f) => f.replace('.json', ''))
  const read = <T>(rel: string): T => JSON.parse(readFileSync(resolve(dir, rel), 'utf8'))

  it.each(ids)('%s: overlay counts are the fixed-threshold TP/FP/FN/TN for every model and seed', (id) => {
    const tiles = read<MarsTilesFile>(`tiles/${id}.json`)
    const site = read<MarsSiteFile>(`sites/${id}.json`)
    for (const [key, model] of [['cnn', 'CNN'], ['qcnn', 'CNN_QCNN']] as const) {
      for (const seed of tiles.seeds) {
        const c = countOutcomes(tiles, key, String(seed))
        const row = site.per_seed.find((r) => r.regime === 'fixed' && r.model === model && r.seed === seed)!
        expect([c['safe-agree'], c['safe-wrong'], c['unsafe-wrong'], c['unsafe-agree']]).toEqual([row.TP, row.FP, row.FN, row.TN])
      }
    }
  })

  it.each(ids)('%s: tiles sit on a grid of their own size', (id) => {
    const t = read<MarsTilesFile>(`tiles/${id}.json`)
    expect(t.row.every((r) => r % t.size_native === 0) && t.col.every((c) => c % t.size_native === 0)).toBe(true)
  })
})
