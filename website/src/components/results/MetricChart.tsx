import { useState } from 'react'
import type { Metric } from '../../data/types'
import { AnimatedNumber } from '../AnimatedNumber'
import { formatMetric, scaleMax, type MetricDef } from './metrics'

/** A model as charts show it; `data` selects its colour (CSS [data-model]). */
export interface ChartModel {
  key: string
  label: string
  short: string
  data: string
}

export interface Point {
  id: string
  label: string
  value: Metric
  active: boolean
}

export interface Row {
  model: ChartModel
  mean: Metric
  sd: number | null
  points: Point[]
  /** Smallest and largest single value (shown as text; never an interval). */
  range?: { min: Metric; max: Metric }
}

interface Props {
  def: MetricDef
  rows: Row[]
  pointNoun: string
  /** A reference value drawn as a hairline across every row (e.g. the majority baseline). */
  reference?: { value: number; label: string }
}

/**
 * One metric for every model, in the order given (fixed order; never sorted by value): a bar
 * for the mean, dots for the single runs, the mean ± SD and the range as text, and an
 * optional reference line. Hover shows the exact value of a bar or dot.
 */
export function MetricChart({ def, rows, pointNoun, reference }: Props) {
  const max = scaleMax(def, [...rows.flatMap((r) => [r.mean, ...r.points.map((p) => p.value)]), reference?.value ?? null])
  const frac = (v: Metric) => (v === null ? 0 : Math.max(0, Math.min(1, v / max)))
  const [hover, setHover] = useState<{ row: number; text: string; x: number } | null>(null)
  const fmtSd = (sd: number) => (def.percent ? (sd * 100).toFixed(def.digits) : sd.toFixed(def.digits))

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
              {reference && (
                <span className="metric-ref-pos" style={{ transform: `translateX(${frac(reference.value) * 100}%)` }}>
                  <span className="metric-ref" />
                </span>
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
              {r.sd !== null && <span className="metric-sd num"> ± {fmtSd(r.sd)}</span>}
              {r.range && r.range.min !== null && r.range.max !== null && (
                <span className="metric-range num">
                  {formatMetric(def, r.range.min)} to {formatMetric(def, r.range.max)}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
      {reference && (
        <p className="metric-ref-label">
          <span className="metric-ref-key" aria-hidden="true" /> {reference.label}: <span className="num">{formatMetric(def, reference.value)}</span>
        </p>
      )}
      <p className="visually-hidden">Bars show the mean; dots show each {pointNoun}.</p>
    </div>
  )
}
