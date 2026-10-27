import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { MotionContext, useMotion } from './motionContext'

const STORAGE_KEY = 'mme.motion-paused'
const REDUCE_QUERY = '(prefers-reduced-motion: reduce)'

function readPaused(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function prefersReduced(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia(REDUCE_QUERY).matches
}

export function MotionProvider({ children }: { children: ReactNode }) {
  const [reduced, setReduced] = useState(prefersReduced)
  const [paused, setPausedState] = useState(readPaused)

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mq = window.matchMedia(REDUCE_QUERY)
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const setPaused = useCallback((next: boolean) => {
    setPausedState(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, next ? '1' : '0')
    } catch {
      // Storage unavailable (private mode); the choice lasts for this visit only.
    }
  }, [])

  const value = useMemo(() => ({ allowed: !reduced && !paused, reduced, paused, setPaused }), [reduced, paused, setPaused])
  return <MotionContext.Provider value={value}>{children}</MotionContext.Provider>
}

/** The visible pause control required for ambient loops longer than five seconds. */
export function MotionToggle({ className }: { className?: string }) {
  const { reduced, paused, setPaused } = useMotion()
  if (reduced) return null
  return (
    <button type="button" className={className} aria-pressed={paused} onClick={() => setPaused(!paused)}>
      {paused ? 'Play motion' : 'Pause motion'}
    </button>
  )
}
