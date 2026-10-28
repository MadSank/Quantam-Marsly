import { MotionConfig } from 'motion/react'
import { Link, Navigate, Route, Routes, useLocation } from 'react-router'
import { MotionProvider } from './components/motion'
import { Landing } from './pages/Landing'
import { MarsExplorer } from './pages/MarsExplorer'
import { MarsGlobe } from './pages/MarsGlobe'
import { WorldLayout } from './theme/WorldLayout'
import type { World } from './theme/world'

// Placeholder pages; replaced by the real views in build steps 9–11 (PLAN.md).

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

// Explorer links from before the globe view (step 8b) carried their state on /mars; send any
// that hold explorer parameters on to /mars/explore with every parameter kept.
const EXPLORER_PARAMS = ['model', 'seed', 'truth', 'scope']

function MarsEntry() {
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  if (EXPLORER_PARAMS.some((k) => params.has(k))) return <Navigate to={{ pathname: '/mars/explore', search }} replace />
  return <MarsGlobe />
}

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <MotionProvider>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/mars" element={<MarsEntry />} />
          <Route path="/mars/explore" element={<MarsExplorer />} />
          <Route path="/moon" element={<Explorer world="moon" />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MotionProvider>
    </MotionConfig>
  )
}
