import { useEffect, useRef } from 'react'
import { Link } from 'react-router'
import sitesFile from '../../public/data/mars/sites.json'
import moonSummary from '../../public/data/moon/summary.json'
import imageManifest from '../../public/data/img/manifest.json'
import { Globe } from '../components/globe/Globe'
import type { GlobeSpec } from '../components/globe/spec'
import { MotionToggle } from '../components/motion'
import { useMotion } from '../components/motionContext'
import { StarField } from '../components/StarField'
import type { MarsSitesFile, MoonSummary } from '../data/types'
import { dataUrl } from '../lib/dataUrl'
import type { World } from '../theme/world'
import '../styles/landing.css'

const fmt = new Intl.NumberFormat('en-GB')

const mars = sitesFile as MarsSitesFile
const moon = moonSummary as MoonSummary
const marsTiles = mars.sites.reduce((n, s) => n + s.n_tiles, 0)
const globes = imageManifest.globe as unknown as Record<World, GlobeSpec>
const moonMosaicPerClass = imageManifest.landing.moon.sample_ids.length / moon.classes.length

interface Door {
  world: World
  name: string
  line: string
  figures: Array<[number, string]>
  image: { src: string; width: number; height: number; alt: string }
}

// Fixed order, Mars then Moon; both doors carry the same weight.
const DOORS: Door[] = [
  {
    world: 'mars',
    name: 'Mars',
    line: 'HiRISE terrain tiles with slope, roughness and hazard targets.',
    figures: [
      [mars.sites.length, 'sites'],
      [marsTiles, 'tiles'],
    ],
    image: {
      ...imageManifest.landing.mars,
      alt: `HiRISE orthoimage ${mars.sites[0].ortho_product_id}, ${mars.sites[0].name}`,
    },
  },
  {
    world: 'moon',
    name: 'Moon',
    line: 'LROCNet test crops labelled ejecta, oldcrater or none.',
    figures: [
      [moon.split_sizes.test, 'test crops'],
      [moon.classes.length, 'classes'],
    ],
    image: {
      ...imageManifest.landing.moon,
      alt: `A mosaic of LROCNet test crops, ${moonMosaicPerClass} from each class`,
    },
  },
]

// Pointer parallax: the sky and planets shift a few pixels against the pointer, at two depths.
const DEPTH_STARS = 6
const DEPTH_PLANETS = 14

function useParallax(enabled: boolean) {
  const stars = useRef<HTMLDivElement>(null)
  const planets = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const layers: Array<[HTMLDivElement | null, number]> = [
      [stars.current, DEPTH_STARS],
      [planets.current, DEPTH_PLANETS],
    ]
    const place = (dx: number, dy: number) => {
      for (const [el, depth] of layers) if (el) el.style.transform = `translate3d(${dx * depth}px, ${dy * depth}px, 0)`
    }
    if (!enabled) {
      place(0, 0)
      return
    }
    let frame = 0
    let target = [0, 0]
    const onMove = (e: PointerEvent) => {
      target = [(e.clientX / window.innerWidth - 0.5) * -2, (e.clientY / window.innerHeight - 0.5) * -2]
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0
        place(target[0], target[1])
      })
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => {
      window.removeEventListener('pointermove', onMove)
      cancelAnimationFrame(frame)
    }
  }, [enabled])

  return { stars, planets }
}

export function Landing() {
  const { allowed } = useMotion()
  const { stars, planets } = useParallax(allowed)

  return (
    <div className="landing" data-motion={allowed ? 'on' : 'off'}>
      <div className="landing-sky" aria-hidden="true">
        <div ref={stars} className="sky-layer">
          <StarField className="stars" />
        </div>
        <div ref={planets} className="sky-layer">
          <Globe spec={globes.mars} className="planet planet--mars" />
          <Globe spec={globes.moon} className="planet planet--moon" />
        </div>
      </div>

      <header className="landing-nav container">
        <Link to="/" className="wordmark">
          Mars and Moon explorer
        </Link>
        <nav aria-label="Site">
          <MotionToggle className="motion-toggle" />
          <Link to="/about">About the study</Link>
        </nav>
      </header>

      <main className="landing-main">
        <section className="landing-hero container" aria-labelledby="landing-title">
          <div className="landing-head">
            <h1 id="landing-title">Choose a world.</h1>
            <p className="lead">A CNN and a hybrid CNN-QCNN, compared on Mars and Moon imagery.</p>
          </div>

          <ul className="doors" aria-label="Worlds">
            {DOORS.map((door) => (
              <li key={door.world} className="door-float" data-world={door.world}>
                <Link to={`/${door.world}`} className="door" data-world={door.world}>
                  <figure className="door-figure">
                    <img
                      src={dataUrl(door.image.src)}
                      width={door.image.width}
                      height={door.image.height}
                      alt={door.image.alt}
                      fetchPriority="high"
                    />
                  </figure>
                  <div className="door-body">
                    <h2>{door.name}</h2>
                    <p className="door-figures">
                      {door.figures.map(([value, unit]) => (
                        <span key={unit} className="door-figure-item">
                          <span className="num">{fmt.format(value)}</span> <span className="unit">{unit}</span>
                        </span>
                      ))}
                    </p>
                    <p className="door-line">{door.line}</p>
                    <span className="door-action">Explore {door.name}</span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      </main>

      <footer className="landing-foot container">
        <p>
          Results are mixed, and no quantum advantage is claimed. <Link to="/about">Read the caveats</Link>
        </p>
        <p className="credit">
          Cards: HiRISE {mars.sites[0].ortho_product_id} (NASA/JPL/University of Arizona) and LROCNet. Globes: Viking
          MDIM 2.1 (NASA Ames, USGS) and LROC WAC colour (NASA SVS).
        </p>
      </footer>
    </div>
  )
}
