import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import caveatsFile from '../../public/data/caveats.json'
import imageManifest from '../../public/data/img/manifest.json'
import reliefFile from '../../public/data/img/globe/moon_relief.json'
import summaryFile from '../../public/data/moon/summary.json'
import { FlatOnError } from '../components/globeview/FlatOnError'
import { flatMapRect, viewFor, type GlobeView } from '../components/globeview/projection'
import { useGlobeControls } from '../components/globeview/useGlobeControls'
import { useSize } from '../components/globeview/useSize'
import { webglAvailable } from '../components/globeview/webgl'
import { CropSprite } from '../components/moon/CropSprite'
import { CLASS_NAMES, LOCATION_NOTE, MOON_CLASSES } from '../components/moon/moonData'
import { MotionToggle } from '../components/motion'
import { useMotion } from '../components/motionContext'
import { StarField } from '../components/StarField'
import { WorldHeader } from '../components/WorldChrome'
import type { CaveatsFile, MoonClass, MoonClassFile, MoonSummary } from '../data/types'
import { dataUrl } from '../lib/dataUrl'
import { useJsonAll } from '../lib/useJson'
import { WorldLayout } from '../theme/WorldLayout'
import '../styles/explorer.css'
import '../styles/globe.css'
import '../styles/moon.css'

const GlobeSphere = lazy(() => import('../components/globeview/GlobeSphere'))

const summary = summaryFile as unknown as MoonSummary
const colourTexture = (imageManifest.globe as unknown as Record<'moon', { textures: Record<'2k', { src: string }>; credit: string }>).moon
const caveats = (caveatsFile as CaveatsFile).caveats
const FOOTER_CAVEATS = ['moon_reproducibility', 'moon_provenance', 'no_quantum_advantage'].map((id) => caveats.find((c) => c.id === id)!)
const fmt = new Intl.NumberFormat('en-GB')

// The near side, face on.
const OVERVIEW: GlobeView = viewFor(0, 0, 1)
const SAMPLES = 4 // crops per card: the first four of the class by file name (a fixed rule, not a selection)
const CARD_CROP = 64
// The cards sit on the right; the flat map leaves them room.
const MAP_INSETS = { top: 120, right: 388, bottom: 96, left: 24 }

function ClassCards({ files, current }: { files: MoonClassFile[] | null; current: MoonClass | null }) {
  return (
    <ol className="class-cards" aria-label="Crater classes">
      {MOON_CLASSES.map((cls, i) => {
        const file = files?.[i]
        const n = summary.split_counts.test[cls]
        return (
          <li key={cls} className="class-card" data-class={cls} data-current={cls === current || undefined}>
            <div className="class-card-head">
              <h2>{CLASS_NAMES[cls]}</h2>
              <span className="class-card-count">
                <span className="num">{fmt.format(n)}</span> test crops
              </span>
            </div>
            <div className="class-card-crops">
              {file?.sprite &&
                file.crops
                  .slice(0, SAMPLES)
                  .map((c) => <CropSprite key={c.sample_id} sprite={file.sprite!} index={c.img} size={CARD_CROP} alt={`${CLASS_NAMES[cls]} test crop ${c.sample_id.split('/').pop()}`} />)}
            </div>
            <p className="class-card-note">First four test crops by file name.</p>
            <Link className="button-primary" to={{ pathname: '/moon/explore', search: `?class=${cls}` }}>
              Explore class
            </Link>
          </li>
        )
      })}
    </ol>
  )
}

export function MoonGlobe() {
  const [params, setParams] = useSearchParams()
  const current = (MOON_CLASSES as string[]).includes(params.get('class') ?? '') ? (params.get('class') as MoonClass) : null
  const { allowed, reduced } = useMotion()
  const [webgl] = useState(webglAvailable)
  const [failed, setFailed] = useState(false)
  const flat = !webgl || failed || params.get('view') === 'map'
  const [stage, size] = useSize<HTMLDivElement>()
  const [ready, setReady] = useState(false)
  const { data: files } = useJsonAll<MoonClassFile>(MOON_CLASSES.map((c) => `moon/classes/${c}.json`))
  const controls = useGlobeControls({ initial: OVERVIEW, size, reduced, autoRotate: allowed && !flat })

  const setView = useCallback(
    (map: boolean) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          if (map) next.set('view', 'map')
          else next.delete('view')
          return next
        },
        { replace: true },
      ),
    [setParams],
  )

  // Wheel zoom needs a non-passive listener.
  useEffect(() => {
    const el = stage.current
    if (!el || flat) return
    el.addEventListener('wheel', controls.onWheel, { passive: false })
    return () => el.removeEventListener('wheel', controls.onWheel)
  }, [stage, flat, controls.onWheel])

  const measured = size.width > 0 && size.height > 0
  const rect = flatMapRect(size, size.width < 900 ? { top: 16, right: 16, bottom: 16, left: 16 } : MAP_INSETS)

  return (
    <WorldLayout world="moon">
      <div className="globe-page" data-mode={flat ? 'map' : 'globe'}>
        <div className="globe-sky" aria-hidden="true">
          <StarField className="globe-stars" />
        </div>
        <WorldHeader world="moon" />
        <main className="globe-main">
          <h1 className="visually-hidden">Moon crater classes</h1>
          <div
            ref={stage}
            className="globe-stage"
            role="region"
            aria-roledescription={flat ? 'map' : 'globe'}
            aria-label={flat ? 'Flat map of the Moon' : 'Globe of the Moon'}
            aria-describedby="moon-globe-help"
            tabIndex={0}
            data-ready={ready || flat || undefined}
            onKeyDown={(e) => {
              if (!flat) controls.onKeyDown(e)
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
            <p id="moon-globe-help" className="globe-help">
              {flat ? 'The flat map has no pins: the crops have no verified locations.' : 'Drag to turn the Moon; scroll or pinch to zoom. Arrow keys turn it, + and − zoom.'}
            </p>
          </div>
          <p className="moon-globe-note">{LOCATION_NOTE}</p>
          <ClassCards files={files} current={current} />
          <div className="moon-globe-side">
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
                  onClick={() => setView(key === 'map')}
                >
                  {label}
                </button>
              ))}
            </div>
            {(!webgl || failed) && <p className="globe-note">This browser cannot draw the 3D globe, so the flat map is shown.</p>}
            <MotionToggle className="motion-toggle" />
          </div>
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
