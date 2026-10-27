// Internal consistency of the exported data (independent of the audit's reported values).
import { describe, expect, it } from 'vitest'
import type {
  MarsSiteFile,
  MarsSitesFile,
  MarsTilesFile,
  MoonClassFile,
  MoonSummary,
} from '../data/types'
import { loadData } from './data'

const { sites } = loadData<MarsSitesFile>('mars/sites.json')
const moon = loadData<MoonSummary>('moon/summary.json')

describe('Mars tiles', () => {
  it('site_01 has 2,195 tiles', () => {
    expect(sites.find((s) => s.site_id === 'site_01')?.n_tiles).toBe(2195)
  })

  it.each(sites.map((s) => [s.site_id, s] as const))('%s: every column has one value per tile', (_, site) => {
    const t = loadData<MarsTilesFile>(`mars/tiles/${site.site_id}.json`)
    const n = site.n_tiles
    expect(t.tile_id).toHaveLength(n)
    expect(new Set(t.tile_id).size).toBe(n)
    for (const col of [t.row, t.col, t.true.slope, t.true.rough, t.true.hazard, t.true.safe]) {
      expect(col).toHaveLength(n)
    }
    for (const model of ['cnn', 'qcnn'] as const) {
      expect(Object.keys(t.pred[model]).sort()).toEqual(t.seeds.map(String).sort())
      for (const seed of t.seeds) {
        const p = t.pred[model][String(seed)]
        for (const col of [p.slope, p.rough, p.hazard, p.safe]) expect(col).toHaveLength(n)
      }
    }
    expect(t.true.safe.reduce((a, b) => a + b, 0)).toBe(site.n_true_safe)
  })

  // Recompute per-seed metrics from the tile file and compare with the exported metrics file.
  it.each(sites.map((s) => [s.site_id, s] as const))('%s: tile predictions reproduce per-seed metrics', (_, site) => {
    const t = loadData<MarsTilesFile>(`mars/tiles/${site.site_id}.json`)
    const metrics = loadData<MarsSiteFile>(`mars/sites/${site.site_id}.json`)
    for (const [key, model] of [['cnn', 'CNN'], ['qcnn', 'CNN_QCNN']] as const) {
      for (const seed of t.seeds) {
        const p = t.pred[key][String(seed)]
        const row = metrics.per_seed.find((r) => r.regime === 'fixed' && r.model === model && r.seed === seed)
        expect(row).toBeDefined()
        let tp = 0, fp = 0, tn = 0, fn = 0
        p.safe.forEach((ps, i) => {
          const ts = t.true.safe[i]
          if (ps && ts) tp++
          else if (ps && !ts) fp++
          else if (!ps && !ts) tn++
          else fn++
        })
        expect([tp, fp, tn, fn]).toEqual([row!.TP, row!.FP, row!.TN, row!.FN])
        const sq = p.slope.reduce<number>((acc, v, i) => acc + ((v as number) - (t.true.slope[i] as number)) ** 2, 0)
        const rmse = Math.sqrt(sq / p.slope.length)
        // Inputs are rounded to 3 dp, so allow a small difference.
        expect(Math.abs(rmse - (row!.RMSE_mean_slope_5m as number))).toBeLessThan(2e-3)
      }
    }
  })
})

describe('Moon crops', () => {
  it.each(moon.classes)('%s: crops, labels and confidences are consistent', (cls) => {
    const doc = loadData<MoonClassFile>(`moon/classes/${cls}.json`)
    expect(doc.crops).toHaveLength(moon.split_counts.test[cls])
    for (const crop of doc.crops) {
      expect(crop.truth).toBe(cls)
      for (const model of moon.models) {
        for (const seed of doc.seeds) {
          const p = crop.pred[model][String(seed)]
          expect(p.conf).toBeGreaterThanOrEqual(0)
          expect(p.conf).toBeLessThanOrEqual(1)
          expect(Math.abs(p.probs.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(5e-4)
          const argmax = p.probs.indexOf(Math.max(...p.probs))
          // Probabilities are rounded to 4 dp; a near-tie can flip the rounded argmax.
          const top = p.probs[moon.classes.indexOf(p.label)]
          expect(top).toBeGreaterThanOrEqual(Math.max(...p.probs) - 1e-4)
          if (argmax !== moon.classes.indexOf(p.label)) expect(Math.max(...p.probs) - top).toBeLessThanOrEqual(1e-4)
          expect(Math.abs(p.conf - Math.max(...p.probs))).toBeLessThanOrEqual(1e-4)
        }
      }
    }
  })

  it('779 test crops in total', () => {
    const n = moon.classes.reduce(
      (acc, cls) => acc + loadData<MoonClassFile>(`moon/classes/${cls}.json`).crops.length,
      0,
    )
    expect(n).toBe(779)
  })
})
