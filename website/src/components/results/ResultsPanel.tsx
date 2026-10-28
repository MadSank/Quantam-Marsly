import { useId, useState } from 'react'
import caveatsFile from '../../../public/data/caveats.json'
import headlineFile from '../../../public/data/mars/headline.json'
import type { CaveatsFile, MarsHeadline, MarsModel, MarsSite, MarsSiteFile, Metric, Regime } from '../../data/types'
import { useJson, useJsonAll } from '../../lib/useJson'
import { AnimatedNumber } from '../AnimatedNumber'
import { DECISIONS, formatMetric, MARS_MODELS, REGIMES, REGRESSION, scaleMax, type MetricDef } from './metrics'

const headline = headlineFile as unknown as MarsHeadline
const caveats = (caveatsFile as CaveatsFile).caveats
const mixedResults = caveats.find((c) => c.id === 'mars_mixed_results')!
// Every caveat that applies on Mars, in the order caveats.json lists them.
const marsCaveats = caveats.filter((c) => c.world === 'mars' || c.world === 'both')

export type Scope = 'site' | 'all'

interface Point {
  id: string
  label: string
  value: Metric
  active: boolean
}

interface Row {
  model: (typeof MARS_MODELS)[number]
  mean: Metric
  sd: number | null
  points: Point[]
}

const num = (v: unknown): Metric => (typeof v === 'number' && Number.isFinite(v) ? v : null)

// ---------------------------------------------------------------- one metric, both models

function MetricChart({ def, rows, pointNoun }: { def: MetricDef; rows: Row[]; pointNoun: string }) {
  const max = scaleMax(def, rows.flatMap((r) => [r.mean, ...r.points.map((p) => p.value)]))
  const frac = (v: Metric) => (v === null ? 0 : Math.max(0, Math.min(1, v / max)))
  const [hover, setHover] = useState<{ row: number; text: string; x: number } | null>(null)

  return (
    <div className="metric" data-metric={def.key}>
      <div className="metric-head">
        <h4>{def.label}</h4>
        <span className="metric-direction">{def.direction ?? def.note}</span>
      </div>
      <div className="metric-rows" onPointerLeave={() => setHover(null)}>
        {rows.map((r, i) => (
          <div key={r.model.key} className="metric-row" data-model={r.model.data}>
            <span className="metric-model">{r.model.short}</span>
            <span className="metric-track">
              {r.mean !== null && (
                <span
                  className="metric-bar"
                  style={{ clipPath: `inset(0 ${(1 - frac(r.mean)) * 100}% 0 0 round 0 4px 4px 0)` }}
                  onPointerEnter={() =>
                    setHover({ row: i, text: `${r.model.label}, mean: ${formatMetric(def, r.mean)}`, x: frac(r.mean) })
                  }
                />
              )}
              {r.points.map((p) => (
                <span key={p.id} className="metric-dot-pos" style={{ transform: `translateX(${frac(p.value) * 100}%)` }}>
                  <span
                    className="metric-dot"
                    data-active={p.active || undefined}
                    hidden={p.value === null}
                    onPointerEnter={() =>
                      setHover({ row: i, text: `${r.model.label}, ${p.label}: ${formatMetric(def, p.value)}`, x: frac(p.value) })
                    }
                  />
                </span>
              ))}
              {hover?.row === i && (
                <span className="metric-tip" role="tooltip" style={{ transform: `translateX(${hover.x * 100}%)` }}>
                  <span>{hover.text}</span>
                </span>
              )}
            </span>
            <span className="metric-value">
              <AnimatedNumber className="num" value={r.mean} format={(v) => formatMetric(def, v)} />
              {r.sd !== null && (
                <span className="metric-sd num">
                  {' '}± {def.percent ? (r.sd * 100).toFixed(def.digits) : r.sd.toFixed(def.digits)}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
      <p className="visually-hidden">
        Bars show the mean; dots show each {pointNoun}.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------- table view

function TableView({ title, defs, regime, rowsFor, columns }: {
  title: string
  defs: MetricDef[]
  regime: Regime
  rowsFor: (def: MetricDef, regime: Regime) => Row[]
  columns: Array<{ id: string; label: string }>
}) {
  return (
    <details className="table-view">
      <summary>Table view: {title}</summary>
      <div className="table-scroll">
        <table>
          <caption className="visually-hidden">{title}</caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col">Model</th>
              {columns.map((c) => (
                <th key={c.id} scope="col">
                  {c.label}
                </th>
              ))}
              <th scope="col">Mean</th>
            </tr>
          </thead>
          <tbody>
            {defs.flatMap((def) =>
              rowsFor(def, regime).map((r, i) => (
                <tr key={`${def.key}-${r.model.key}`}>
                  {i === 0 && (
                    <th scope="rowgroup" rowSpan={2}>
                      {def.label}
                    </th>
                  )}
                  <th scope="row">{r.model.short}</th>
                  {columns.map((c) => {
                    const p = r.points.find((x) => x.id === c.id)
                    return (
                      <td key={c.id} className="num" data-active={p?.active || undefined}>
                        {p ? formatMetric(def, p.value) : ''}
                      </td>
                    )
                  })}
                  <td className="num">{formatMetric(def, r.mean)}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </details>
  )
}

// ---------------------------------------------------------------- panel

interface PanelProps {
  site: MarsSite
  sites: MarsSite[]
  seed: string
  scope: Scope
  onScope: (s: Scope) => void
}

/**
 * Per-site (or five-site) results for both models, in fixed order: regression error, then the
 * safe-proxy decisions under both regimes, with recall, false-positive rate and coverage together.
 * The mixed-results caveat stays pinned under the scrolling content.
 */
export function ResultsPanel({ site, sites, seed, scope, onScope }: PanelProps) {
  const titleId = useId()
  const siteFile = useJson<MarsSiteFile>(`mars/sites/${site.site_id}.json`)
  const allFiles = useJsonAll<MarsSiteFile>(scope === 'all' ? sites.map((s) => `mars/sites/${s.site_id}.json`) : null)

  const ready = scope === 'site' ? siteFile.data : allFiles.data
  const error = scope === 'site' ? siteFile.error : allFiles.error

  const rowsFor = (def: MetricDef, regime: Regime): Row[] =>
    MARS_MODELS.map((model) => {
      if (scope === 'site') {
        const f = siteFile.data!
        const points = f.per_seed
          .filter((r) => r.regime === regime && r.model === model.key)
          .map((r) => ({ id: String(r.seed), label: `seed ${r.seed}`, value: num(r[def.key]), active: String(r.seed) === seed }))
        return { model, mean: num(f.mean[regime][model.key as MarsModel][def.key]), sd: null, points }
      }
      const cell = headline[regime][model.key][def.key]
      const points = allFiles.data!.map((f, i) => ({
        id: f.site_id,
        label: sites[i].name,
        value: num(f.mean[regime][model.key][def.key]),
        active: f.site_id === site.site_id,
      }))
      return { model, mean: num(cell?.mean), sd: num(cell?.sd), points }
    })

  const columns =
    scope === 'site'
      ? (siteFile.data?.per_seed ?? [])
          .filter((r) => r.regime === 'fixed' && r.model === 'CNN')
          .map((r) => ({ id: String(r.seed), label: String(r.seed) }))
      : sites.map((s, i) => ({ id: s.site_id, label: `S${i + 1}` }))
  const pointNoun = scope === 'site' ? 'seed' : 'site'

  return (
    <aside className="results" aria-labelledby={titleId}>
      <div className="results-scroll">
        <header className="results-head">
          <div className="results-title">
            <h2 id={titleId}>Results</h2>
            <div className="scope-toggle" role="radiogroup" aria-label="Scope">
              {(
                [
                  ['site', 'This site'],
                  ['all', 'All five sites'],
                ] as const
              ).map(([key, label]) => (
                <button key={key} type="button" role="radio" aria-checked={scope === key} onClick={() => onScope(key)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          <p className="results-sub">
            {scope === 'site'
              ? `Test site ${site.name}. Bars are the mean of three seeds; dots are single seeds.`
              : 'All five test sites. Bars are the mean across sites, ± sample SD; dots are single sites.'}
          </p>
          <ul className="results-legend" aria-label="Models">
            {MARS_MODELS.map((m) => (
              <li key={m.key} data-model={m.data}>
                <span className="swatch swatch--fill" aria-hidden="true" />
                {m.label}
              </li>
            ))}
          </ul>
        </header>

        {error ? (
          <p className="results-status" role="alert">
            Results could not be loaded. Reload the page to try again.
          </p>
        ) : !ready ? (
          <p className="results-status">Loading results…</p>
        ) : (
          <div key={`${site.site_id}-${scope}`} className="results-body">
            <section className="metric-group" aria-labelledby={`${titleId}-reg`}>
              <h3 id={`${titleId}-reg`}>Terrain regression error</h3>
              {REGRESSION.map((def) => (
                <MetricChart key={def.key} def={def} rows={rowsFor(def, 'fixed')} pointNoun={pointNoun} />
              ))}
              <TableView title="Terrain regression error" defs={REGRESSION} regime="fixed" rowsFor={rowsFor} columns={columns} />
            </section>

            {REGIMES.map((regime) => (
              <section key={regime.key} className="metric-group" aria-labelledby={`${titleId}-${regime.key}`}>
                <h3 id={`${titleId}-${regime.key}`}>Safe-proxy decisions, {regime.label.toLowerCase()}</h3>
                <p className="metric-group-note">{regime.note}</p>
                {DECISIONS.map((def) => (
                  <MetricChart key={def.key} def={def} rows={rowsFor(def, regime.key)} pointNoun={pointNoun} />
                ))}
                <TableView
                  title={`Safe-proxy decisions, ${regime.label.toLowerCase()}`}
                  defs={DECISIONS}
                  regime={regime.key}
                  rowsFor={rowsFor}
                  columns={columns}
                />
              </section>
            ))}
          </div>
        )}
        <details className="caveat-list">
          <summary>All Mars caveats ({marsCaveats.length})</summary>
          <ul>
            {marsCaveats.map((c) => (
              <li key={c.id} data-caveat={c.id}>
                <strong>{c.title}.</strong> {c.text}
              </li>
            ))}
          </ul>
        </details>
      </div>
      <p className="results-caveat" data-caveat={mixedResults.id}>
        <strong>{mixedResults.title}.</strong> {mixedResults.text}
      </p>
    </aside>
  )
}
