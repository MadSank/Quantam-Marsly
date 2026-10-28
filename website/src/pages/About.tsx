import { Link } from 'react-router'
import aboutFile from '../../public/data/about.json'
import caveatsFile from '../../public/data/caveats.json'
import summaryFile from '../../public/data/moon/summary.json'
import { MOON_MODELS } from '../components/moon/moonData'
import type { AboutFile, Caveat, CaveatsFile, MoonSummary } from '../data/types'
import '../styles/about.css'

const about = aboutFile as unknown as AboutFile
const caveats = (caveatsFile as CaveatsFile).caveats
const summary = summaryFile as unknown as MoonSummary
const timingCaveat = caveats.find((c) => c.id === summary.timing.caveat_id)!

const WORLD_LABEL: Record<Caveat['world'], string> = { mars: 'Mars', moon: 'Moon', both: 'Both worlds' }
const GROUPS: Array<Caveat['world']> = ['mars', 'moon', 'both']

// What each Moon timing measures (experiments/moon/train.py, inference_timing): the CNN runs the
// whole image on the GPU; the two heads run on embeddings the frozen encoder already computed, on the CPU.
const TIMED: Record<string, string> = {
  cnn: 'Whole image through encoder and classifier, GPU',
  hybrid_qcnn: 'Head only, on precomputed embeddings, CPU (simulated circuit)',
  frozen_embedding_mlp: 'Head only, on precomputed embeddings, CPU',
}

function meanSd(values: number[]) {
  const m = values.reduce((a, b) => a + b, 0) / values.length
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1))
  return { m, sd }
}

const SECTIONS = [
  { id: 'method', label: 'Method' },
  { id: 'timings', label: 'Timings' },
  { id: 'caveats', label: 'Caveats' },
  { id: 'sources', label: 'Data sources' },
]

/** The neutral shared page: method, the separately measured timings, every caveat, and the data sources. */
export function About() {
  return (
    <div className="about">
      <header className="about-header">
        <Link to="/" className="wordmark">
          Mars and Moon explorer
        </Link>
        <nav aria-label="Site" className="about-nav">
          <Link to="/mars">Mars</Link>
          <Link to="/moon">Moon</Link>
          <Link to="/about" aria-current="page">
            About
          </Link>
        </nav>
      </header>

      <div className="about-layout container">
        <nav className="about-toc" aria-label="On this page">
          <ol>
            {SECTIONS.map((s) => (
              <li key={s.id}>
                {/* Hash routing owns the URL fragment, so in-page jumps scroll and move focus by script. */}
                <a
                  href="#/about"
                  onClick={(e) => {
                    e.preventDefault()
                    const heading = document.getElementById(`${s.id}-title`)
                    heading?.scrollIntoView({ block: 'start' })
                    heading?.focus({ preventScroll: true })
                  }}
                >
                  {s.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <main className="about-main">
          <h1>About the study</h1>
          <p className="about-lead">
            A CNN and a hybrid CNN-QCNN, compared on Mars terrain and Moon craters, with what was done, what the figures
            can and cannot support, and where the data comes from.
          </p>

          <section id="method" className="about-section" aria-labelledby="method-title">
            <h2 id="method-title" tabIndex={-1}>Method</h2>
            {about.method.map((m) => (
              <article key={m.id} className="about-block" data-topic={m.world} data-method={m.id}>
                <h3>{m.title}</h3>
                {m.paragraphs.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
              </article>
            ))}
          </section>

          <section id="timings" className="about-section" aria-labelledby="timings-title">
            <h2 id="timings-title" tabIndex={-1}>Timings</h2>
            <p className="about-note" data-caveat={timingCaveat.id}>
              <strong>Separately measured, not comparable.</strong> {timingCaveat.text}
            </p>
            <div className="about-table-scroll">
              <table className="about-table" data-comparable="false">
                <caption>
                  Moon inference time per test crop, milliseconds, mean ± sample SD over {summary.seeds.length} seeds. Each row
                  times something different.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">What is timed</th>
                    <th scope="col">ms per crop</th>
                  </tr>
                </thead>
                <tbody>
                  {MOON_MODELS.map((m) => {
                    const { m: mean, sd } = meanSd(Object.values(summary.timing.milliseconds_per_image[m.key]))
                    return (
                      <tr key={m.key}>
                        <th scope="row">{m.label}</th>
                        <td>{TIMED[m.key]}</td>
                        <td className="num">
                          {mean.toFixed(3)} ± {sd.toFixed(3)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section id="caveats" className="about-section" aria-labelledby="caveats-title">
            <h2 id="caveats-title" tabIndex={-1}>Caveats</h2>
            <p className="about-section-lead">
              Every caveat is quoted word for word from its source file. The explorers show the same text beside the
              results it qualifies.
            </p>
            {GROUPS.map((g) => (
              <div key={g} className="caveat-group">
                <h3>{WORLD_LABEL[g]}</h3>
                <ul>
                  {caveats
                    .filter((c) => c.world === g)
                    .map((c) => (
                      <li key={c.id} data-caveat={c.id}>
                        <p className="caveat-title">{c.title}</p>
                        <p>{c.text}</p>
                        <p className="caveat-source">
                          Source: <span className="mono">{c.source}</span>
                        </p>
                      </li>
                    ))}
                </ul>
              </div>
            ))}
          </section>

          <section id="sources" className="about-section" aria-labelledby="sources-title">
            <h2 id="sources-title" tabIndex={-1}>Data sources</h2>
            <ul className="source-list">
              {about.sources.map((s) => (
                <li key={s.name} className="source" data-source={s.name}>
                  <h3>{s.name}</h3>
                  <dl>
                    <div>
                      <dt>Provider</dt>
                      <dd>{s.provider}</dd>
                    </div>
                    <div>
                      <dt>Used for</dt>
                      <dd>{s.used_for}</dd>
                    </div>
                    <div>
                      <dt>Licence</dt>
                      <dd>
                        {s.licence} <a href={s.licence_url}>Terms</a>
                      </dd>
                    </div>
                    <div>
                      <dt>Credit</dt>
                      <dd>{s.credit}</dd>
                    </div>
                    <div>
                      <dt>Citation</dt>
                      <dd className="source-citation">{s.citation}</dd>
                    </div>
                    <div>
                      <dt>Where</dt>
                      <dd>
                        <a href={s.url} className="source-url">
                          {s.url.replace(/^https?:\/\//, '')}
                        </a>
                      </dd>
                    </div>
                  </dl>
                </li>
              ))}
            </ul>
          </section>
        </main>
      </div>

      <footer className="about-footer container">
        <p>No quantum advantage claimed; circuits simulated.</p>
        <p>
          <Link to="/mars">Explore Mars</Link> <Link to="/moon">Explore the Moon</Link>
        </p>
      </footer>
    </div>
  )
}
