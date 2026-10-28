import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Link, useSearchParams } from 'react-router'
import caveatsFile from '../../public/data/caveats.json'
import summaryFile from '../../public/data/moon/summary.json'
import { CropSprite } from '../components/moon/CropSprite'
import { CLASS_NAMES, CLASS_TAGS, modelFromUrl, MOON_CLASSES, MOON_MODELS } from '../components/moon/moonData'
import { MoonResultsPanel } from '../components/moon/MoonResultsPanel'
import { Rail } from '../components/Rail'
import { WorldFooter, WorldHeader } from '../components/WorldChrome'
import type { CaveatsFile, MoonClass, MoonClassFile, MoonCrop, MoonSummary } from '../data/types'
import { useJson } from '../lib/useJson'
import { WorldLayout } from '../theme/WorldLayout'
import '../styles/explorer.css'
import '../styles/moon.css'

const summary = summaryFile as unknown as MoonSummary
const fmt = new Intl.NumberFormat('en-GB')
const CELL = 88 // crop size in the mosaic, CSS px
const provenance = (caveatsFile as CaveatsFile).caveats.find((c) => c.id === 'moon_provenance')!

const fileName = (sampleId: string) => sampleId.split('/').pop() ?? sampleId

// ---------------------------------------------------------------- controls

function MoonControls({ model, seed, seeds, labels, onModel, onSeed, onLabels }: {
  model: string
  seed: string
  seeds: string[]
  labels: boolean
  onModel: (url: string) => void
  onSeed: (s: string) => void
  onLabels: (on: boolean) => void
}) {
  const seedId = useId()
  return (
    <div className="overlay-controls">
      <div className="model-toggle" role="radiogroup" aria-label="Model">
        {MOON_MODELS.map((m) => (
          <button key={m.key} type="button" role="radio" aria-checked={m.url === model} data-model={m.data} onClick={() => onModel(m.url)}>
            <span className="swatch swatch--fill" aria-hidden="true" />
            {m.label}
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
        <input type="checkbox" role="switch" checked={labels} onChange={(e) => onLabels(e.target.checked)} />
        Predicted class
      </label>
    </div>
  )
}

// ---------------------------------------------------------------- legend

function MoonLegend({ cls, file, error, modelKey, seed }: {
  cls: MoonClass
  file: MoonClassFile | null
  error: boolean
  modelKey: (typeof MOON_MODELS)[number]['key']
  seed: string
}) {
  const model = MOON_MODELS.find((m) => m.key === modelKey)!
  const counts = useMemo(() => {
    const c: Record<MoonClass, number> = { ejecta: 0, oldcrater: 0, none: 0 }
    file?.crops.forEach((crop) => {
      const p = crop.pred[modelKey]?.[seed]
      if (p) c[p.label]++
    })
    return c
  }, [file, modelKey, seed])
  return (
    <section className="legend" aria-labelledby="moon-legend-title" data-model={model.data}>
      <h2 id="moon-legend-title">Mosaic</h2>
      <p className="legend-lead">
        {model.label}, seed <span className="num">{seed}</span>. Every crop here is labelled {CLASS_TAGS[cls]}.
      </p>
      {error ? (
        <p className="legend-status" role="alert">
          The crops could not be loaded. Reload the page to try again.
        </p>
      ) : !file ? (
        <p className="legend-status">Loading crops…</p>
      ) : null}
      <ul className="moon-legend-rows">
        <li>
          <span className="crop-key" data-correct aria-hidden="true" />
          Solid outline: predicted {CLASS_TAGS[cls]} (right)
        </li>
        <li>
          <span className="crop-key" aria-hidden="true" />
          Dashed outline: predicted another class (wrong)
        </li>
      </ul>
      <table className="legend-matrix moon-counts">
        <caption className="visually-hidden">Predictions for the crops in this class</caption>
        <thead>
          <tr>
            <th scope="col">Predicted as</th>
            <th scope="col">Crops</th>
          </tr>
        </thead>
        <tbody>
          {MOON_CLASSES.map((c) => (
            <tr key={c} data-predicted={c}>
              <th scope="row">
                {CLASS_TAGS[c]}
                {c === cls ? ' (right)' : ''}
              </th>
              <td className="num">{file ? fmt.format(counts[c]) : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

// ---------------------------------------------------------------- mosaic

function Mosaic({ file, cls, modelKey, seed, labels, selected, onSelect }: {
  file: MoonClassFile
  cls: MoonClass
  modelKey: (typeof MOON_MODELS)[number]['key']
  seed: string
  labels: boolean
  selected: number | null
  onSelect: (i: number) => void
}) {
  const grid = useRef<HTMLDivElement>(null)
  const [focusIndex, setFocusIndex] = useState(0)
  const model = MOON_MODELS.find((m) => m.key === modelKey)!

  // One tab stop for the whole mosaic; arrows, Home and End move between crops (roving tabindex).
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = file.crops.length
    const cols = Math.max(1, Math.floor((grid.current?.clientWidth ?? CELL) / (CELL + 6)))
    const moves: Record<string, number> = {
      ArrowRight: focusIndex + 1,
      ArrowLeft: focusIndex - 1,
      ArrowDown: focusIndex + cols,
      ArrowUp: focusIndex - cols,
      Home: 0,
      End: n - 1,
    }
    if (!(e.key in moves)) return
    e.preventDefault()
    const next = Math.max(0, Math.min(n - 1, moves[e.key]))
    setFocusIndex(next)
    grid.current?.querySelector<HTMLButtonElement>(`[data-index="${next}"]`)?.focus()
  }

  return (
    <div ref={grid} className="mosaic" role="group" aria-label={`${CLASS_NAMES[cls]} test crops`} data-model={model.data} onKeyDown={onKeyDown}>
      {file.crops.map((crop, i) => {
        const p = crop.pred[modelKey][seed]
        const correct = p.label === crop.truth
        return (
          <button
            key={crop.sample_id}
            type="button"
            className="crop"
            data-index={i}
            data-correct={correct || undefined}
            aria-pressed={selected === i}
            tabIndex={i === focusIndex ? 0 : -1}
            aria-label={`${fileName(crop.sample_id)}: predicted ${CLASS_TAGS[p.label]}, ${correct ? 'right' : 'wrong'}`}
            onClick={() => {
              setFocusIndex(i)
              onSelect(i)
            }}
          >
            {file.sprite && <CropSprite sprite={file.sprite} index={crop.img} size={CELL} />}
            {labels && <span className="crop-tag">{CLASS_TAGS[p.label]}</span>}
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------- inspector

function CropInspector({ crop, file, seed, onClose }: { crop: MoonCrop; file: MoonClassFile; seed: string; onClose: () => void }) {
  const id = useId()
  return (
    <section className="inspector moon-inspector" aria-labelledby={id} aria-live="polite">
      <div className="inspector-head">
        <h2 id={id}>
          Crop <span className="mono">{fileName(crop.sample_id)}</span>
        </h2>
        <button type="button" className="inspector-close" onClick={onClose}>
          Close
        </button>
      </div>
      <div className="moon-inspector-body">
        {file.sprite && <CropSprite sprite={file.sprite} index={crop.img} size={112} alt={`Test crop ${fileName(crop.sample_id)}`} />}
        <table className="inspector-table">
          <caption className="visually-hidden">
            Predictions for this crop, seed {seed}. True class: {CLASS_TAGS[crop.truth]}.
          </caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Predicted</th>
              {file.classes.map((c) => (
                <th key={c} scope="col">
                  p({CLASS_TAGS[c]})
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOON_MODELS.map((m) => {
              const p = crop.pred[m.key][seed]
              return (
                <tr key={m.key} data-model={m.data}>
                  <th scope="row">{m.short}</th>
                  <td>{CLASS_TAGS[p.label]}</td>
                  {p.probs.map((v, k) => (
                    <td key={k} className="num">
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="inspector-foot">
        True class: {CLASS_TAGS[crop.truth]}. Seed <span className="num">{seed}</span>; probabilities are the softmax of each
        model's saved logits.
      </p>
    </section>
  )
}

// ---------------------------------------------------------------- page

export function MoonExplorer() {
  const [params, setParams] = useSearchParams()
  const cls: MoonClass = (MOON_CLASSES as string[]).includes(params.get('class') ?? '') ? (params.get('class') as MoonClass) : 'ejecta'
  const model = modelFromUrl(params.get('model'))
  const seeds = summary.seeds.map(String)
  const seed = seeds.includes(params.get('seed') ?? '') ? params.get('seed')! : seeds[0]
  const labels = params.get('labels') === '1'
  const { data: file, error } = useJson<MoonClassFile>(`moon/classes/${cls}.json`)
  const [picked, setPicked] = useState<{ cls: MoonClass; index: number } | null>(null)
  const selected = file && picked?.cls === cls ? picked.index : null

  // Escape closes the crop inspector.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setPicked(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const setParam = (key: string, value: string | null) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (value === null) next.delete(key)
        else next.set(key, value)
        return next
      },
      { replace: true },
    )
  const to = (id: string) => {
    const next = new URLSearchParams(params)
    next.set('class', id)
    return { search: `?${next}` }
  }
  const total = summary.split_sizes.test
  const n = summary.split_counts.test[cls]

  return (
    <WorldLayout world="moon">
      <div className="explorer moon-explorer">
        <WorldHeader world="moon" />
        <section className="readout" aria-label="Class readout">
          <h1 className="readout-name">{CLASS_NAMES[cls]}</h1>
          <dl className="readout-facts">
            <div>
              <dt>Test crops</dt>
              <dd className="num">
                {fmt.format(n)} of {fmt.format(total)} ({((100 * n) / total).toFixed(1)}%)
              </dd>
            </div>
            <div>
              <dt>Seeds</dt>
              <dd className="num">
                {summary.seeds.length} ({summary.seeds[0]}–{summary.seeds[summary.seeds.length - 1]})
              </dd>
            </div>
            <div className="readout-product">
              <dt>Split</dt>
              <dd>supplied LROCNet test split</dd>
            </div>
          </dl>
          <Link className="details-toggle" to={{ pathname: '/moon', search: `?class=${cls}` }}>
            Back to the Moon
          </Link>
        </section>
        <main className="explorer-main">
          <div className="sidebar">
            <Rail
              label="Classes"
              selected={cls}
              to={to}
              items={MOON_CLASSES.map((c) => ({
                id: c,
                label: CLASS_NAMES[c],
                detail: `${fmt.format(summary.split_counts.test[c])} test crops`,
              }))}
            />
            <MoonLegend cls={cls} file={file} error={error} modelKey={model.key} seed={seed} />
          </div>
          <section className="image-panel moon-panel" aria-label={`${CLASS_NAMES[cls]} crops`}>
            <div className="moon-panel-scroll">
              {file && (
                <Mosaic
                  key={cls}
                  file={file}
                  cls={cls}
                  modelKey={model.key}
                  seed={seed}
                  labels={labels}
                  selected={selected}
                  onSelect={(index) => setPicked({ cls, index })}
                />
              )}
              <p className="moon-provenance">
                <strong>{provenance.title}.</strong> {provenance.text}
              </p>
            </div>
            <div className="panzoom-controls">
              <MoonControls
                model={model.url}
                seed={seed}
                seeds={seeds}
                labels={labels}
                onModel={(m) => setParam('model', m)}
                onSeed={(s) => setParam('seed', s)}
                onLabels={(on) => setParam('labels', on ? '1' : null)}
              />
            </div>
            {file && selected !== null && (
              <CropInspector crop={file.crops[selected]} file={file} seed={seed} onClose={() => setPicked(null)} />
            )}
          </section>
          <MoonResultsPanel seed={seed} />
        </main>
        <WorldFooter />
      </div>
    </WorldLayout>
  )
}
