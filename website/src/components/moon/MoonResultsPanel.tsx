import { useId } from 'react'
import caveatsFile from '../../../public/data/caveats.json'
import summaryFile from '../../../public/data/moon/summary.json'
import type { CaveatsFile, MoonSummary } from '../../data/types'
import { MetricChart, type Row } from '../results/MetricChart'
import { formatMetric, num, type MetricDef } from '../results/metrics'
import { MOON_METRICS, MOON_MODELS } from './moonData'

const summary = summaryFile as unknown as MoonSummary
const caveats = (caveatsFile as CaveatsFile).caveats
const reproducibility = caveats.find((c) => c.id === 'moon_reproducibility')!
// Every caveat that applies on the Moon, in the order caveats.json lists them.
const moonCaveats = caveats.filter((c) => c.world === 'moon' || c.world === 'both')
const baseline = summary.metrics.majority_train_baseline

function rowsFor(def: MetricDef, seed: string): Row[] {
  return MOON_MODELS.map((model) => {
    const cell = summary.metrics[model.key][def.key]
    const points = summary.per_seed
      .filter((r) => r.model === model.key)
      .map((r) => ({ id: String(r.seed), label: `seed ${r.seed}`, value: num(r[def.key]), active: String(r.seed) === seed }))
    const range = summary.ranges[model.key][def.key as keyof MoonSummary['ranges'][typeof model.key]]
    return { model, mean: num(cell?.mean), sd: num(cell?.sd), points, range: { min: num(range?.min), max: num(range?.max) } }
  })
}

/**
 * Moon results over all 779 test crops from the deterministic 10-seed run (audit Addendum):
 * per model, the mean ± SD, every seed as a dot, the min–max range, and the majority baseline
 * as a reference line. The reproducibility caveat stays pinned beside the metrics.
 */
export function MoonResultsPanel({ seed }: { seed: string }) {
  const titleId = useId()
  const seeds = summary.seeds
  return (
    <aside className="results" aria-labelledby={titleId}>
      <div className="results-scroll">
        <header className="results-head">
          <div className="results-title">
            <h2 id={titleId}>Results</h2>
          </div>
          <p className="results-sub">
            All {summary.split_sizes.test} test crops. Bars are the mean of {seeds.length} deterministic seeds ({seeds[0]}–
            {seeds[seeds.length - 1]}), ± sample SD; dots are single seeds; the small figures are the lowest and highest seed.
          </p>
          <ul className="results-legend" aria-label="Models">
            {MOON_MODELS.map((m) => (
              <li key={m.key} data-model={m.data}>
                <span className="swatch swatch--fill" aria-hidden="true" />
                {m.label}
              </li>
            ))}
          </ul>
        </header>

        <div className="results-body">
          <section className="metric-group" aria-labelledby={`${titleId}-cls`}>
            <h3 id={`${titleId}-cls`}>Crater classification, all classes</h3>
            {MOON_METRICS.map((def) => (
              <MetricChart
                key={def.key}
                def={def}
                rows={rowsFor(def, seed)}
                pointNoun="seed"
                reference={{ value: baseline[def.key].mean, label: 'Majority baseline' }}
              />
            ))}
            {MOON_METRICS.map((def) => (
              <details key={`t-${def.key}`} className="table-view">
                <summary>Table view: {def.label.toLowerCase()} per seed</summary>
                <div className="table-scroll">
                  <table>
                    <caption className="visually-hidden">{def.label} per seed and model</caption>
                    <thead>
                      <tr>
                        <th scope="col">Seed</th>
                        {MOON_MODELS.map((m) => (
                          <th key={m.key} scope="col">
                            {m.short}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {seeds.map((s) => (
                        <tr key={s}>
                          <th scope="row" className="num">
                            {s}
                          </th>
                          {MOON_MODELS.map((m) => {
                            const r = summary.per_seed.find((p) => p.model === m.key && p.seed === s)
                            return (
                              <td key={m.key} className="num" data-active={String(s) === seed || undefined}>
                                {formatMetric(def, num(r?.[def.key]))}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            ))}
          </section>
        </div>

        <details className="caveat-list">
          <summary>All Moon caveats ({moonCaveats.length})</summary>
          <ul>
            {moonCaveats.map((c) => (
              <li key={c.id} data-caveat={c.id}>
                <strong>{c.title}.</strong> {c.text}
              </li>
            ))}
          </ul>
        </details>
      </div>
      <p className="results-caveat" data-caveat={reproducibility.id}>
        <strong>{reproducibility.title}.</strong> {reproducibility.text}
      </p>
    </aside>
  )
}
