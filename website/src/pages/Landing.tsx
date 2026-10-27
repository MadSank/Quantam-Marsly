import { Link } from 'react-router'
import sitesFile from '../../public/data/mars/sites.json'
import moonSummary from '../../public/data/moon/summary.json'
import imageManifest from '../../public/data/img/manifest.json'
import type { MarsSitesFile, MoonSummary } from '../data/types'
import type { World } from '../theme/world'
import '../styles/landing.css'

const dataUrl = (rel: string) => `${import.meta.env.BASE_URL}data/${rel}`
const fmt = new Intl.NumberFormat('en-GB')

const mars = sitesFile as MarsSitesFile
const moon = moonSummary as MoonSummary
const marsTiles = mars.sites.reduce((n, s) => n + s.n_tiles, 0)

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
    line: 'HiRISE strips cut into terrain tiles, with slope, roughness and hazard targets.',
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
      alt: 'A mosaic of LROCNet test crops, eight from each class',
    },
  },
]

export function Landing() {
  return (
    <div className="landing">
      <header className="landing-nav container">
        <Link to="/" className="wordmark">
          Mars and Moon explorer
        </Link>
        <nav aria-label="Site">
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
              <li key={door.world}>
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
                    <p className="door-line">{door.line}</p>
                    <p className="door-figures">
                      {door.figures.map(([value, unit]) => (
                        <span key={unit} className="door-figure-item">
                          <span className="num">{fmt.format(value)}</span> <span className="unit">{unit}</span>
                        </span>
                      ))}
                    </p>
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
          Imagery: HiRISE {mars.sites[0].ortho_product_id} (NASA/JPL/University of Arizona) and LROCNet crater crops.
        </p>
      </footer>
    </div>
  )
}
