import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'
import sitesFile from '../../public/data/mars/sites.json'
import imageManifest from '../../public/data/img/manifest.json'
import { Atmosphere } from '../components/Atmosphere'
import { Globe } from '../components/globe/Globe'
import type { GlobeSpec } from '../components/globe/spec'
import type { ModelKey } from '../components/overlay/draw'
import { tileAt } from '../components/overlay/draw'
import { OverlayControls, OverlayLegend, TileInspector, TileOverlay } from '../components/overlay/Overlay'
import { niceScaleBar } from '../components/panzoom/math'
import { PanZoomViewport } from '../components/panzoom/PanZoomViewport'
import { Rail } from '../components/Rail'
import { WorldFooter, WorldHeader } from '../components/WorldChrome'
import type { MarsSite, MarsSitesFile, MarsTilesFile } from '../data/types'
import { dataUrl } from '../lib/dataUrl'
import { useJson } from '../lib/useJson'
import { WorldLayout } from '../theme/WorldLayout'
import '../styles/explorer.css'

const sites = (sitesFile as MarsSitesFile).sites
const stretch = imageManifest.mars as unknown as Record<string, { display_stretch: { percentiles: [number, number] } }>
const marsGlobe = (imageManifest.globe as unknown as Record<'mars', GlobeSpec>).mars
const fmt = new Intl.NumberFormat('en-GB')

// Zoom stops at twice the preview's own pixel density, so a downsampled preview is never blown up further.
const OVERZOOM = 2

function latLabel(lat: number) {
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}`
}

function lonLabel(lon: number) {
  return `${lon.toFixed(2)}° E`
}

function formatLength(m: number) {
  return m >= 1000 ? `${fmt.format(m / 1000)} km` : `${fmt.format(m)} m`
}

/** Read ?site= and fall back to the first site; other params (model, seed) are kept as they are. */
function useSite(): [MarsSite, URLSearchParams] {
  const [params] = useSearchParams()
  const site = sites.find((s) => s.site_id === params.get('site')) ?? sites[0]
  return [site, params]
}

function StripImage({ site }: { site: MarsSite }) {
  const [loaded, setLoaded] = useState(false)
  const size = { width: site.native_width_px, height: site.native_height_px }
  return (
    <>
      {site.thumb && <img className="strip-placeholder" src={dataUrl(site.thumb.src)} alt="" style={size} draggable={false} />}
      {site.preview && (
        <img
          className="strip-image"
          src={dataUrl(site.preview.src)}
          width={site.preview.width}
          height={site.preview.height}
          alt={`HiRISE orthoimage ${site.ortho_product_id}, ${site.name}`}
          style={size}
          draggable={false}
          decoding="async"
          data-loaded={loaded || undefined}
          onLoad={() => setLoaded(true)}
        />
      )}
    </>
  )
}

/** The mission readout under the nav: the compact fact strip that is always shown. */
function Readout({ site, detailsOpen, onToggle, triggerRef, drawerId }: {
  site: MarsSite
  detailsOpen: boolean
  onToggle: () => void
  triggerRef: React.RefObject<HTMLButtonElement | null>
  drawerId: string
}) {
  return (
    <section className="readout" aria-label="Site readout">
      <h1 className="readout-name">{site.name}</h1>
      <dl className="readout-facts">
        <div>
          <dt>Centre</dt>
          <dd className="mono">
            <span>{latLabel(site.center.lat)}</span> <span>{lonLabel(site.center.lon_e)}</span>
          </dd>
        </div>
        <div>
          <dt>Tiles</dt>
          <dd className="num">{fmt.format(site.n_tiles)}</dd>
        </div>
        <div className="readout-product">
          <dt>Orthoimage</dt>
          <dd className="mono">{site.ortho_product_id}</dd>
        </div>
      </dl>
      <button
        ref={triggerRef}
        type="button"
        className="details-toggle"
        aria-expanded={detailsOpen}
        aria-controls={drawerId}
        aria-keyshortcuts="i"
        onClick={onToggle}
      >
        Site details <kbd aria-hidden="true">i</kbd>
      </button>
    </section>
  )
}

/** Full site facts in a non-modal drawer over the strip; Esc or Close returns focus to the trigger. */
function DetailsDrawer({ site, open, onClose, id, drawerRef }: {
  site: MarsSite
  open: boolean
  onClose: () => void
  id: string
  drawerRef: React.RefObject<HTMLElement | null>
}) {
  const p = stretch[site.site_id]?.display_stretch.percentiles
  const rows: Array<[string, string, 'num' | 'mono']> = [
    ['Tiles', fmt.format(site.n_tiles), 'num'],
    ['Held out in', `fold ${site.fold}`, 'num'],
    ['Centre', `${latLabel(site.center.lat)}, ${lonLabel(site.center.lon_e)}`, 'mono'],
    ['Ortho pixel', `${site.native_resolution_m.toFixed(2)} m`, 'num'],
    ['Strip size', `${fmt.format(site.native_width_px)} × ${fmt.format(site.native_height_px)} px`, 'num'],
    ['Orthoimage', site.ortho_product_id, 'mono'],
    ['DTM', site.dtm_product_id, 'mono'],
  ]
  return (
    <aside
      ref={drawerRef}
      id={id}
      className="drawer"
      role="dialog"
      aria-modal="false"
      aria-labelledby={`${id}-title`}
      tabIndex={-1}
      data-open={open || undefined}
      inert={!open}
    >
      <div className="drawer-head">
        <h2 id={`${id}-title`}>Site details</h2>
        <button type="button" className="drawer-close" onClick={onClose}>
          Close
        </button>
      </div>
      <dl className="facts-list">
        {rows.map(([k, v, kind]) => (
          <div key={k} className={kind === 'mono' && v.length > 24 ? 'facts-row-id' : undefined}>
            <dt>{k}</dt>
            <dd className={kind}>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="facts-note">
        Each site is one leave-one-site-out fold: the models were trained on three other sites, validated on a fourth and
        tested on this one.
      </p>
      {p && (
        <p className="facts-note">
          Display stretch only ({p[0]}–{p[1]} percentile); the models saw the unstretched ortho tiles.
        </p>
      )}
    </aside>
  )
}

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
}

export function MarsExplorer() {
  const [site, params] = useSite()
  const [, setParams] = useSearchParams()
  const to = (id: string) => {
    const next = new URLSearchParams(params)
    next.set('site', id)
    return { search: `?${next}` }
  }

  // Overlay state lives in the URL: ?model=cnn|qcnn&seed=42&truth=1.
  const { data: tiles, error: tilesError } = useJson<MarsTilesFile>(`mars/tiles/${site.site_id}.json`)
  const seeds = (tiles?.seeds ?? [42, 43, 44]).map(String)
  const model: ModelKey = params.get('model') === 'qcnn' ? 'qcnn' : 'cnn'
  const seed = seeds.includes(params.get('seed') ?? '') ? params.get('seed')! : seeds[0]
  const showTruth = params.get('truth') === '1'
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

  // The inspected tile belongs to one site; switching site clears it.
  const [picked, setPicked] = useState<{ site: string; index: number } | null>(null)
  const selected = tiles && picked?.site === site.site_id ? picked.index : null
  const pickAt = (point: { x: number; y: number }) => {
    if (!tiles) return
    const index = tileAt(tiles, point.x, point.y)
    setPicked(index === null ? null : { site: site.site_id, index })
  }
  const maxScale = OVERZOOM * (site.preview?.scale_from_native.x ?? 1)
  const target = useMemo(() => ({ lat: site.center.lat, lon: site.center.lon_e }), [site])

  const drawerId = useId()
  const [detailsOpen, setDetailsOpen] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const drawer = useRef<HTMLElement>(null)

  const close = useCallback(() => {
    setDetailsOpen(false)
    if (drawer.current?.contains(document.activeElement)) trigger.current?.focus()
  }, [])
  const toggle = useCallback(() => (detailsOpen ? close() : setDetailsOpen(true)), [detailsOpen, close])

  // Opening moves focus into the drawer so its content is read next.
  useEffect(() => {
    if (detailsOpen) drawer.current?.focus()
  }, [detailsOpen])

  // i toggles the drawer from anywhere on the page; Esc closes it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || isTyping(e.target)) return
      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault()
        toggle()
      } else if (e.key === 'Escape' && detailsOpen) {
        e.preventDefault()
        setDetailsOpen(false)
        trigger.current?.focus()
      } else if (e.key === 'Escape') {
        setPicked(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggle, detailsOpen])

  return (
    <WorldLayout world="mars">
      <div className="explorer">
        <Atmosphere siteId={site.site_id} />
        <WorldHeader world="mars" />
        <Readout site={site} detailsOpen={detailsOpen} onToggle={toggle} triggerRef={trigger} drawerId={drawerId} />
        <main className="explorer-main">
          <div className="sidebar">
            <Rail
              label="Sites"
              selected={site.site_id}
              to={to}
              items={sites.map((s) => ({
                id: s.site_id,
                label: s.name,
                detail: `${fmt.format(s.n_tiles)} tiles`,
                thumb: s.thumb && { ...s.thumb, alt: '' },
              }))}
            />
            <OverlayLegend tiles={tiles} error={tilesError} model={model} seed={seed} showTruth={showTruth} />
          </div>
          <section className="image-panel" aria-label={`${site.name} strip`}>
            <PanZoomViewport
              key={site.site_id}
              content={{ width: site.native_width_px, height: site.native_height_px }}
              maxScale={maxScale}
              label={`${site.name} orthoimage strip`}
              onTap={pickAt}
              onKey={(e, centre) => {
                if (e.key !== 'Enter') return false
                pickAt(centre)
                return true
              }}
              toolbar={
                <OverlayControls
                  model={model}
                  seed={seed}
                  seeds={seeds}
                  showTruth={showTruth}
                  onModel={(m) => setParam('model', m)}
                  onSeed={(s) => setParam('seed', s)}
                  onTruth={(on) => setParam('truth', on ? '1' : null)}
                />
              }
              hud={(view, viewport) => {
                const bar = niceScaleBar(view.scale, site.native_resolution_m)
                return (
                  <>
                    {tiles && (
                      <TileOverlay
                        tiles={tiles}
                        model={model}
                        seed={seed}
                        view={view}
                        viewport={viewport}
                        showTruth={showTruth}
                        selected={selected}
                      />
                    )}
                    <div className="hud-foot">
                      <div className="scale-bar" aria-hidden="true">
                        <span className="scale-bar-line" style={{ width: bar.px }} />
                        <span className="num">{formatLength(bar.length)}</span>
                      </div>
                      <p className="hud-note">Display stretch only</p>
                    </div>
                  </>
                )
              }}
            >
              <StripImage site={site} />
            </PanZoomViewport>
            <div className="locator">
              <Globe spec={marsGlobe} resolution="1k" target={target} pinToken="--pin" className="locator-globe" />
            </div>
            {tiles && selected !== null && (
              <TileInspector tiles={tiles} index={selected} seed={seed} model={model} onClose={() => setPicked(null)} />
            )}
            <DetailsDrawer site={site} open={detailsOpen} onClose={close} id={drawerId} drawerRef={drawer} />
          </section>
        </main>
        <WorldFooter />
      </div>
    </WorldLayout>
  )
}
