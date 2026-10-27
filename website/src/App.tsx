import { MotionConfig } from 'motion/react'
import { Link, Navigate, Route, Routes } from 'react-router'
import { WorldLayout } from './theme/WorldLayout'
import type { World } from './theme/world'

// Placeholder pages; replaced by the real views in build steps 5–10 (PLAN.md).
function Landing() {
  return (
    <main>
      <h1>Choose a world</h1>
      <nav aria-label="Worlds">
        <Link to="/mars">Mars</Link> · <Link to="/moon">Moon</Link> · <Link to="/about">About</Link>
      </nav>
    </main>
  )
}

function Explorer({ world }: { world: World }) {
  return (
    <WorldLayout world={world}>
      <main>
        <h1>{world === 'mars' ? 'Mars' : 'Moon'} explorer</h1>
        <Link to="/">Back to worlds</Link>
      </main>
    </WorldLayout>
  )
}

function About() {
  return (
    <main>
      <h1>About this study</h1>
      <Link to="/">Back to worlds</Link>
    </main>
  )
}

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/mars" element={<Explorer world="mars" />} />
        <Route path="/moon" element={<Explorer world="moon" />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </MotionConfig>
  )
}
