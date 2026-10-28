import { useId, useState } from 'react'

export interface CurveSeries {
  key: string
  label: string
  /** CSS [data-model] value that picks the series colour. */
  data: string
  epoch: number[]
  values: number[]
}

interface Props {
  title: string
  /** "lower is better" or "higher is better", shown beside the title. */
  direction: string
  series: CurveSeries[]
  format: (v: number) => string
}

const W = 560
const H = 240
const PAD = { top: 12, right: 64, bottom: 28, left: 48 }

function niceTicks(lo: number, hi: number, n = 4) {
  const span = hi - lo || 1
  const step0 = span / n
  const mag = 10 ** Math.floor(Math.log10(step0))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0)!
  const start = Math.floor(lo / step) * step
  const ticks: number[] = []
  for (let v = start; v <= hi + step * 1e-6; v += step) ticks.push(Number(v.toFixed(10)))
  if (ticks.at(-1)! < hi) ticks.push(ticks.at(-1)! + step)
  return ticks
}

/**
 * One measure per epoch for each model, as lines on one axis. Models keep their fixed order and
 * colour; each line ends in a direct label, and hover shows every model's value at that epoch.
 */
export function CurveChart({ title, direction, series, format }: Props) {
  const titleId = useId()
  const [hover, setHover] = useState<number | null>(null)
  const maxEpoch = Math.max(...series.flatMap((s) => s.epoch))
  const all = series.flatMap((s) => s.values)
  const ticks = niceTicks(Math.min(...all), Math.max(...all))
  const lo = ticks[0]
  const hi = ticks.at(-1)!
  const x = (e: number) => PAD.left + ((e - 1) / Math.max(1, maxEpoch - 1)) * (W - PAD.left - PAD.right)
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo || 1)) * (H - PAD.top - PAD.bottom)
  const epochs = Array.from({ length: maxEpoch }, (_, i) => i + 1)

  // Direct labels at line ends, nudged apart so they never overlap.
  const ends = series
    .map((s) => ({ key: s.key, label: s.label, data: s.data, y: y(s.values.at(-1)!), x: x(s.epoch.at(-1)!) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14)

  return (
    <figure className="curve" aria-labelledby={titleId}>
      <figcaption className="curve-head">
        <span id={titleId} className="curve-title">
          {title}
        </span>
        <span className="curve-direction">{direction}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="curve-plot" role="img" aria-label={`${title} per epoch; values in the table view below`} onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t} className="curve-grid">
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {format(t)}
            </text>
          </g>
        ))}
        {epochs.map((e) => (
          <text key={e} className="curve-axis" x={x(e)} y={H - 8} textAnchor="middle">
            {e}
          </text>
        ))}
        {hover !== null && <line className="curve-cross" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} />}
        {series.map((s) => (
          <g key={s.key} data-model={s.data} className="curve-series">
            <polyline points={s.epoch.map((e, i) => `${x(e)},${y(s.values[i])}`).join(' ')} />
            {s.epoch.map((e, i) => (
              <circle key={e} cx={x(e)} cy={y(s.values[i])} r={hover === e ? 4.5 : 3} />
            ))}
          </g>
        ))}
        {ends.map((l) => (
          <text key={l.key} className="curve-end" x={l.x + 8} y={l.y} dy="0.32em">
            {l.label}
          </text>
        ))}
        {epochs.map((e) => (
          <rect
            key={e}
            className="curve-hit"
            x={x(e) - (W - PAD.left - PAD.right) / Math.max(1, maxEpoch - 1) / 2}
            y={PAD.top}
            width={(W - PAD.left - PAD.right) / Math.max(1, maxEpoch - 1)}
            height={H - PAD.top - PAD.bottom}
            onPointerEnter={() => setHover(e)}
          />
        ))}
      </svg>
      {/* Pointer readout only; the table below carries the same values for every reader. */}
      <p className="curve-readout" aria-hidden="true">
        {hover === null ? (
          'Epoch on the horizontal axis. Point at the chart for values.'
        ) : (
          <>
            Epoch {hover}:{' '}
            {series.map((s) => {
              const i = s.epoch.indexOf(hover)
              return (
                <span key={s.key} data-model={s.data} className="curve-value">
                  {s.label} {i < 0 ? 'stopped' : format(s.values[i])}
                </span>
              )
            })}
          </>
        )}
      </p>
      <details className="curve-table">
        <summary>Table view: {title.toLowerCase()} per epoch</summary>
        <div className="curve-table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Epoch</th>
                {series.map((s) => (
                  <th key={s.key} scope="col">
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {epochs.map((e) => (
                <tr key={e}>
                  <th scope="row">{e}</th>
                  {series.map((s) => {
                    const i = s.epoch.indexOf(e)
                    return (
                      <td key={s.key} className="num">
                        {i < 0 ? 'stopped' : format(s.values[i])}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  )
}
