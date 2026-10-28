import { Link } from 'react-router'
import { WORLDS, type World } from '../theme/world'

const NAMES: Record<World, string> = { mars: 'Mars', moon: 'Moon' }

/** Header for both explorers: wordmark, world switch and About. */
export function WorldHeader({ world }: { world: World }) {
  return (
    <header className="world-header">
      <Link to="/" className="wordmark">
        Mars and Moon explorer
      </Link>
      <nav aria-label="Site" className="world-nav">
        <ul className="world-switch" aria-label="Worlds">
          {WORLDS.map((w) => (
            <li key={w}>
              <Link to={`/${w}`} aria-current={w === world ? 'page' : undefined}>
                {NAMES[w]}
              </Link>
            </li>
          ))}
        </ul>
        <Link to="/about" className="caveat-chip">
          <span>
            Caveats<span className="chip-more"> and method</span>
          </span>
        </Link>
      </nav>
    </header>
  )
}

/** Footer for both explorers; carries the required quantum caveat (PLAN.md, "Rules for presenting the science"). */
export function WorldFooter() {
  return (
    <footer className="world-footer">
      <p>
        No quantum advantage claimed; circuits simulated. <Link to="/about">Read the caveats</Link>
      </p>
    </footer>
  )
}
