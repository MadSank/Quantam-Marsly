import { Component, lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import caveatsFile from '../../public/data/caveats.json'
import imageManifest from '../../public/data/img/manifest.json'
import reliefFile from '../../public/data/img/globe/mars_relief.json'
import sitesFile from '../../public/data/mars/sites.json'
import { Callouts, type Rect } from '../components/marsglobe/Callouts'
import { latLabel, lonLabel, siteDescription } from '../components/marsglobe/format'
import { FLAT_INSETS, flatMapRect, projectEquirect, projectOrtho, viewFor, zoomForFootprint, type GlobeView, type Size } from '../components/marsglobe/projection'
import { useGlobeControls } from '../components/marsglobe/useGlobeControls'
import { MotionToggle } from '../components/motion'
import { useMotion } from '../components/motionContext'
import { StarField } from '../components/StarField'
import { WorldHeader } from '../components/WorldChrome'
import type { CaveatsFile, MarsSite, MarsSitesFile } from '../data/types'
import { dataUrl } from '../lib/dataUrl'
import { WorldLayout } from '../theme/WorldLayout'
import '../styles/explorer.css'
import '../styles/globe.css'

const GlobeSphere = lazy(() => import('../components/marsglobe/GlobeSphere'))

const sites = (sitesFile as MarsSitesFile).sites
const colourTexture = (imageManifest.globe as unknown as Record<'mars', { textures: Record<'2k', { src: string }>; credit: string }>).mars
const caveats = (caveatsFile as CaveatsFile).caveats
const FOOTER_CAVEATS = ['no_quantum_advantage', 'mars_terrain_proxy'].map((id) => caveats.find((c) => c.id === id)!)
const fmt = new Intl.NumberFormat('en-GB')

// Opening view: 20° E, 10° N puts four of the five sites on the near side.
const OVERVIEW: GlobeView = viewFor(10, 20, 1)
const FOOTPRINT_ZOOM = 20

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

/** Falls back to the flat map if the WebGL scene fails to load or render. */
class FlatOnError extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch() {
    this.props.onError()
  }
  render() {
    return this.state.failed ? null : this.props.children
  }
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setSize({ width: el.clientWidth, height: el.clientHeight })
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}

// ---------------------------------------------------------------- site list and card

function SiteList({ selected, onSelect }: { selected: string | null; onSelect: (id: string) => void }) {
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    const buttons = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('button.site-item')]
    const i = buttons.indexOf(document.activeElement as HTMLButtonElement)
    if (i < 0) return
    const next =
      e.key === 'ArrowDown' ? Math.min(buttons.length - 1, i + 1)
        : e.key === 'ArrowUp' ? Math.max(0, i - 1)
          : e.key === 'Home' ? 0
            : e.key === 'End' ? buttons.length - 1
              : null
    if (next === null) return
    e.preventDefault()
    buttons[next].focus()
  }
  return (
    <nav className="site-list" aria-label="Test sites" onKeyDown={onKeyDown}>
      <h2 className="site-list-title">Five HiRISE test sites</h2>
      <ol>
        {sites.map((s, i) => (
          <li key={s.site_id}>
            <button
              type="button"
              className="site-item"
              aria-current={s.site_id === selected ? 'true' : undefined}
              onClick={() => onSelect(s.site_id)}
            >
              <span className="site-index num" aria-hidden="true">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="site-name">{s.name}</span>
              <span className="site-coords mono">
                {latLabel(s.center.lat)} {lonLabel(s.center.lon_e)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  )
}

function SiteCard({ site, onBack, headingRef, cardRef }: {
  site: MarsSite | null
  onBack: () => void
  headingRef: React.RefObject<HTMLHeadingElement | null>
  cardRef: React.RefObject<HTMLElement | null>
}) {
  if (!site) return null
  return (
    <aside ref={cardRef} className="site-card" aria-label="Selected site">
      {site.thumb && (
        <img className="site-card-thumb" src={dataUrl(site.thumb.src)} width={site.thumb.width} height={site.thumb.height} alt={`HiRISE orthoimage ${site.ortho_product_id}, ${site.name}`} />
      )}
      <div className="site-card-body">
        <h2 ref={headingRef} tabIndex={-1}>
          {site.name}
        </h2>
        <p className="site-card-coords mono">
          {latLabel(site.center.lat)} {lonLabel(site.center.lon_e)}
        </p>
        <p className="site-card-tiles">
          <span className="num">{fmt.format(site.n_tiles)}</span> tiles
        </p>
        <p className="site-card-desc">{siteDescription(site)}</p>
        <div className="site-card-actions">
          <Link className="button-primary" to={{ pathname: '/mars/explore', search: `?site=${site.site_id}` }}>
            Explore site
          </Link>
          <button type="button" className="button-quiet" onClick={onBack}>
            Back
          </button>
        </div>
      </div>
    </aside>
  )
}

// ---------------------------------------------------------------- page

export function MarsGlobe() {
  const [params, setParams] = useSearchParams()
  const selected = sites.find((s) => s.site_id === params.get('site')) ?? null
  const { allowed, reduced } = useMotion()
  const [webgl] = useState(webglAvailable)
  const [failed, setFailed] = useState(false)
  const flat = !webgl || failed || params.get('view') === 'map'
  const [stage, size] = useSize<HTMLDivElement>()
  const [ready, setReady] = useState(false)
  const heading = useRef<HTMLHeadingElement>(null)
  const side = useRef<HTMLDivElement>(null)
  const cardRef = useRef<HTMLElement>(null)
  const [avoid, setAvoid] = useState<Rect[]>([])

  const controls = useGlobeControls({ initial: OVERVIEW, size, reduced, autoRotate: allowed && !selected && !flat })
  const { flyTo } = controls

  const setParam = useCallback(
    (key: string, value: string | null) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          if (value === null) next.delete(key)
          else next.set(key, value)
          return next
        },
        { replace: true },
      ),
    [setParams],
  )

  const select = (id: string) => setParam('site', id)
  const back = () => setParam('site', null)

  // Fly to the selected site, or back out to the whole planet.
  const selectedId = selected?.site_id ?? null
  useEffect(() => {
    const s = sites.find((x) => x.site_id === selectedId)
    if (s) flyTo(viewFor(s.center.lat, s.center.lon_e, zoomForFootprint(s.footprint)))
    else flyTo({ ...OVERVIEW, yaw: OVERVIEW.yaw })
  }, [selectedId, flyTo])

  // Choosing a site moves focus to the card, so its actions are next for keyboard users. A site
  // already in the URL on load is not a choice, so focus stays put (safe under StrictMode re-runs).
  const lastSelected = useRef(selectedId)
  useEffect(() => {
    if (selectedId && selectedId !== lastSelected.current) heading.current?.focus()
    lastSelected.current = selectedId
  }, [selectedId])

  // Wheel zoom needs a non-passive listener.
  useEffect(() => {
    const el = stage.current
    if (!el || flat) return
    el.addEventListener('wheel', controls.onWheel, { passive: false })
    return () => el.removeEventListener('wheel', controls.onWheel)
  }, [stage, flat, controls.onWheel])

  // Below 900 px the list and card stack outside the stage, so the map only keeps a margin.
  // Keep-out areas for callout labels: the site list, and the card while it is open (stage pixels).
  useLayoutEffect(() => {
    const st = stage.current?.getBoundingClientRect()
    if (!st) return
    const rel = (el: Element | null): Rect | null => {
      const r = el?.getBoundingClientRect()
      return r && r.width > 0 ? { left: r.left - st.left, top: r.top - st.top, right: r.right - st.left, bottom: r.bottom - st.top } : null
    }
    const next = [rel(side.current), selectedId ? rel(cardRef.current) : null].filter((r): r is Rect => r !== null)
    setAvoid((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next))
  }, [size, selectedId, stage])

  const rect = flatMapRect(size, size.width < 900 ? { top: 16, right: 16, bottom: 16, left: 16 } : FLAT_INSETS)
  const project = flat
    ? (lat: number, lon: number) => projectEquirect(lat, lon, rect)
    : (lat: number, lon: number) => projectOrtho(lat, lon, controls.view, size)
  const measured = size.width > 0 && size.height > 0

  return (
    <WorldLayout world="mars">
      <div className="globe-page" data-mode={flat ? 'map' : 'globe'}>
        <div className="globe-sky" aria-hidden="true">
          <StarField className="globe-stars" />
        </div>
        <WorldHeader world="mars" />
        <main className="globe-main">
          <h1 className="visually-hidden">Mars test sites</h1>
          <div ref={side} className="globe-side">
            <SiteList selected={selectedId} onSelect={select} />
            <div className="view-toggle" role="radiogroup" aria-label="View">
              {(
                [
                  ['globe', 'Globe'],
                  ['map', 'Flat map'],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  role="radio"
                  aria-checked={(key === 'map') === flat}
                  disabled={key === 'globe' && (!webgl || failed)}
                  onClick={() => setParam('view', key === 'map' ? 'map' : null)}
                >
                  {label}
                </button>
              ))}
            </div>
            {(!webgl || failed) && <p className="globe-note">This browser cannot draw the 3D globe, so the flat map is shown.</p>}
            <MotionToggle className="motion-toggle" />
          </div>
          <div
            ref={stage}
            className="globe-stage"
            role="region"
            aria-roledescription={flat ? 'map' : 'globe'}
            aria-label={flat ? 'Flat map of Mars with the test sites' : 'Globe of Mars with the test sites'}
            aria-describedby="globe-help"
            tabIndex={0}
            data-ready={ready || flat || undefined}
            data-close={!flat && controls.view.zoom >= FOOTPRINT_ZOOM ? true : undefined}
            onKeyDown={(e) => {
              if (e.key === 'Escape' && selected) {
                e.preventDefault()
                back()
              } else if (!flat) controls.onKeyDown(e)
            }}
            {...(flat ? {} : controls.handlers)}
          >
            {flat ? (
              <img
                className="flat-map"
                src={dataUrl(colourTexture.textures['2k'].src)}
                alt=""
                style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}
                draggable={false}
              />
            ) : (
              measured && (
                <FlatOnError onError={() => setFailed(true)}>
                  <Suspense fallback={null}>
                    <div className="globe-canvas">
                      <GlobeSphere
                        textureUrl={dataUrl(colourTexture.textures['2k'].src)}
                        normalUrl={dataUrl(reliefFile.textures['1k'].src)}
                        view={controls.view}
                        size={size}
                        onReady={() => setReady(true)}
                      />
                    </div>
                  </Suspense>
                </FlatOnError>
              )
            )}
            {measured && (ready || flat) && (
              <Callouts
                sites={sites}
                project={project}
                selected={selectedId}
                footprints={!flat && controls.view.zoom >= FOOTPRINT_ZOOM}
                layout={flat ? 'flat' : 'globe'}
                avoid={avoid}
                onSelect={select}
              />
            )}
            <p id="globe-help" className="globe-help">
              {flat
                ? 'Choose a site from the list or a pin.'
                : 'Drag to turn the globe; scroll or pinch to zoom. Arrow keys turn it, + and − zoom, Esc goes back.'}
            </p>
          </div>
          <SiteCard site={selected} onBack={back} headingRef={heading} cardRef={cardRef} />
        </main>
        <footer className="globe-footer">
          {FOOTER_CAVEATS.map((c) => (
            <p key={c.id} data-caveat={c.id}>
              {c.text}
            </p>
          ))}
          <p>
            <Link to="/about">Read the caveats</Link>
          </p>
          <p className="globe-credit">
            Globe: {colourTexture.credit}. Relief: {reliefFile.credit}, exaggerated {reliefFile.vertical_exaggeration}×.
          </p>
        </footer>
      </div>
    </WorldLayout>
  )
}
