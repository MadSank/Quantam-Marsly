import { MotionConfig } from 'motion/react'
import { Navigate, Route, Routes, useLocation } from 'react-router'
import { MotionProvider } from './components/motion'
import { About } from './pages/About'
import { Landing } from './pages/Landing'
import { MarsExplorer } from './pages/MarsExplorer'
import { MarsGlobe } from './pages/MarsGlobe'
import { MoonExplorer } from './pages/MoonExplorer'
import { MoonGlobe } from './pages/MoonGlobe'

// Explorer links from before the globe view (step 8b) carried their state on /mars; send any
// that hold explorer parameters on to /mars/explore with every parameter kept.
const EXPLORER_PARAMS = ['model', 'seed', 'truth', 'scope']
// The Moon globe keeps ?class= (to highlight a card); the explorer-only parameters move on.
const MOON_EXPLORER_PARAMS = ['model', 'seed', 'labels']

function MarsEntry() {
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  if (EXPLORER_PARAMS.some((k) => params.has(k))) return <Navigate to={{ pathname: '/mars/explore', search }} replace />
  return <MarsGlobe />
}

function MoonEntry() {
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  if (MOON_EXPLORER_PARAMS.some((k) => params.has(k))) return <Navigate to={{ pathname: '/moon/explore', search }} replace />
  return <MoonGlobe />
}

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <MotionProvider>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/mars" element={<MarsEntry />} />
          <Route path="/mars/explore" element={<MarsExplorer />} />
          <Route path="/moon" element={<MoonEntry />} />
          <Route path="/moon/explore" element={<MoonExplorer />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MotionProvider>
    </MotionConfig>
  )
}
