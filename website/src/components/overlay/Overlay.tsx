import { useId, useLayoutEffect, useRef } from 'react'
import caveatsFile from '../../../public/data/caveats.json'
import type { CaveatsFile, MarsTilesFile } from '../../data/types'
import type { Size, View } from '../panzoom/math'
import { countOutcomes, drawOverlay, type ModelKey } from './draw'
import { MODELS, modelLabel, OUTCOME_ROWS, predictionStyle, type Outcome } from './style'

const safeProxy = (caveatsFile as CaveatsFile).caveats.find((c) => c.id === 'safe_proxy')!

// ---------------------------------------------------------------- canvas

interface TileOverlayProps {
  tiles: MarsTilesFile
  model: ModelKey
  seed: string
  view: View
  viewport: Size
  showTruth: boolean
  selected: number | null
}

/** The prediction (and optional ground-truth) layer, drawn in viewport pixels from the view on screen. */
export function TileOverlay({ tiles, model, seed, view, viewport, showTruth, selected }: TileOverlayProps) {
  const ref = useRef<HTMLCanvasElement>(null)

  useLayoutEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const w = Math.round(viewport.width * dpr)
    const h = Math.round(viewport.height * dpr)
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    const css = getComputedStyle(canvas)
    const token = (name: string) => css.getPropertyValue(name).trim()
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawOverlay(ctx, {
      tiles,
      model,
      seed,
      view,
      viewport,
      showTruth,
      selected,
      colours: {
        model: token(MODELS.find((m) => m.key === model)!.token),
        truthSafe: token('--truth-safe'),
        truthUnsafe: token('--truth-unsafe'),
        focus: token('--focus'),
        ink: token('--bg'),
      },
    })
  }, [tiles, model, seed, view, viewport, showTruth, selected])

  return (
    <canvas
      ref={ref}
      className="tile-overlay"
      style={{ width: viewport.width, height: viewport.height }}
      data-model={model}
      data-seed={seed}
      data-truth={showTruth || undefined}
      aria-hidden="true"
    />
  )
}

// ---------------------------------------------------------------- controls

interface ControlsProps {
  model: ModelKey
  seed: string
  seeds: string[]
  showTruth: boolean
  onModel: (m: ModelKey) => void
  onSeed: (s: string) => void
  onTruth: (on: boolean) => void
}

/** Model toggle (fixed order, CNN first), seed select and ground-truth switch. */
export function OverlayControls({ model, seed, seeds, showTruth, onModel, onSeed, onTruth }: ControlsProps) {
  const seedId = useId()
  return (
    <div className="overlay-controls">
      <div className="model-toggle" role="radiogroup" aria-label="Model">
        {MODELS.map((m) => (
          <button
            key={m.key}
            type="button"
            role="radio"
            aria-checked={m.key === model}
            data-model={m.key}
            aria-label={m.label}
            onClick={() => onModel(m.key)}
          >
            <span className="swatch swatch--fill" aria-hidden="true" />
            <span className="label-long" aria-hidden="true">
              {m.label}
            </span>
            <span className="label-short" aria-hidden="true">
              {m.short}
            </span>
          </button>
        ))}
      </div>
      <label className="seed-select" htmlFor={seedId}>
        Seed
        <select id={seedId} value={seed} onChange={(e) => onSeed(e.target.value)}>
          {seeds.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <label className="truth-toggle">
        <input type="checkbox" role="switch" checked={showTruth} onChange={(e) => onTruth(e.target.checked)} />
        Ground truth
      </label>
    </div>
  )
}

// ---------------------------------------------------------------- legend

interface LegendProps {
  tiles: MarsTilesFile | null
  error: boolean
  model: ModelKey
  seed: string
  showTruth: boolean
}

const fmt = new Intl.NumberFormat('en-GB')

// Rows: the prediction; columns: proxy safe, proxy unsafe.
const MATRIX: Array<{ row: string; cells: [Outcome, Outcome] }> = [
  { row: 'Predicted safe', cells: ['safe-agree', 'safe-wrong'] },
  { row: 'Predicted unsafe', cells: ['unsafe-wrong', 'unsafe-agree'] },
]
const LABELS = Object.fromEntries(OUTCOME_ROWS.map((r) => [r.outcome, r.label])) as Record<Outcome, string>

/** Explains every encoding on the strip and carries the safe-proxy caveat, always visible. */
export function OverlayLegend({ tiles, error, model, seed, showTruth }: LegendProps) {
  const counts = tiles && tiles.pred[model][seed] ? countOutcomes(tiles, model, seed) : null
  return (
    <section className="legend" aria-labelledby="legend-title" data-model={model}>
      <h2 id="legend-title">Overlay</h2>
      <p className="legend-lead">
        {modelLabel(model)}, seed <span className="num">{seed}</span>. Fixed-threshold decisions.
      </p>
      {error ? (
        <p className="legend-status" role="alert">
          Tile predictions could not be loaded. Reload the page to try again.
        </p>
      ) : null}
      {/* A 2 × 2 key: prediction by row, proxy label by column; each cell names itself for screen readers.
          While the predictions load the counts read "…", so the key keeps its height. */}
      <table className="legend-matrix" aria-busy={!tiles && !error}>
        <caption className="visually-hidden">Tile encodings and tile counts</caption>
        <thead>
          <tr>
            <td />
            <th scope="col">Proxy safe</th>
            <th scope="col">Proxy unsafe</th>
          </tr>
        </thead>
        <tbody>
          {MATRIX.map(({ row, cells }) => (
            <tr key={row}>
              <th scope="row">{row}</th>
              {cells.map((outcome) => {
                const s = predictionStyle(outcome)
                return (
                  <td key={outcome} data-outcome={outcome}>
                    <span className="legend-cell">
                      <span
                        className="swatch"
                        data-fill={s.fill || undefined}
                        data-hatch={s.hatch || undefined}
                        data-outline={s.outline}
                        aria-hidden="true"
                      />
                      <span className="visually-hidden legend-label">{LABELS[outcome]}</span>
                      <span className="legend-count num">{counts ? fmt.format(counts[outcome]) : error ? '' : '…'}</span>
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {showTruth && (
        <p className="legend-truth">
          Ground truth, corner mark:{' '}
          <span data-truth="safe">
            <span className="swatch swatch--truth" data-truth="safe" aria-hidden="true" />
            <span className="legend-label">proxy safe</span>
          </span>{' '}
          <span data-truth="unsafe">
            <span className="swatch swatch--truth" data-truth="unsafe" aria-hidden="true" />
            <span className="legend-label">proxy unsafe</span>
          </span>
        </p>
      )}
      <p className="legend-caveat" data-caveat={safeProxy.id}>
        <strong>{safeProxy.title}.</strong> {safeProxy.text}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------- inspector

interface InspectorProps {
  tiles: MarsTilesFile
  index: number
  seed: string
  model: ModelKey
  onClose: () => void
}

const QUANTITIES = [
  { key: 'slope', label: 'Mean slope', unit: '°', digits: 1 },
  { key: 'rough', label: 'Roughness', unit: ' m', digits: 2 },
  { key: 'hazard', label: 'Hazard fraction', unit: '', digits: 3 },
] as const

const value = (v: number | null, digits: number, unit: string) => (v === null ? 'undefined' : `${v.toFixed(digits)}${unit}`)
const safeWord = (v: number) => (v ? 'safe' : 'unsafe')

/** One tile's targets and both models' predictions for the chosen seed, in fixed model order. */
export function TileInspector({ tiles, index, seed, model, onClose }: InspectorProps) {
  const id = useId()
  return (
    <section className="inspector" aria-labelledby={id} aria-live="polite">
      <div className="inspector-head">
        <h2 id={id}>
          Tile <span className="mono">{tiles.tile_id[index]}</span>
        </h2>
        <button type="button" className="inspector-close" onClick={onClose}>
          Close
        </button>
      </div>
      <table className="inspector-table">
        <caption className="visually-hidden">
          Proxy targets and predictions for this tile, seed {seed}
        </caption>
        <thead>
          <tr>
            <th scope="col">Quantity</th>
            <th scope="col">Target</th>
            {MODELS.map((m) => (
              <th key={m.key} scope="col" data-model={m.key} aria-current={m.key === model || undefined}>
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {QUANTITIES.map((q) => (
            <tr key={q.key}>
              <th scope="row">{q.label}</th>
              <td className="num">{value(tiles.true[q.key][index], q.digits, q.unit)}</td>
              {MODELS.map((m) => (
                <td key={m.key} className="num">
                  {value(tiles.pred[m.key][seed][q.key][index], q.digits, q.unit)}
                </td>
              ))}
            </tr>
          ))}
          <tr>
            <th scope="row">Safe (proxy)</th>
            <td>{safeWord(tiles.true.safe[index])}</td>
            {MODELS.map((m) => (
              <td key={m.key}>{safeWord(tiles.pred[m.key][seed].safe[index])}</td>
            ))}
          </tr>
        </tbody>
      </table>
      <p className="inspector-foot">
        Row <span className="num">{fmt.format(tiles.row[index])}</span>, column{' '}
        <span className="num">{fmt.format(tiles.col[index])}</span> in ortho pixels;{' '}
        <span className="num">{tiles.size_native}</span> px square.
      </p>
    </section>
  )
}
