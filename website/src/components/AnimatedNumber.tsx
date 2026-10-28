import { useLayoutEffect, useState } from 'react'
import { useMotion } from './motionContext'

const DURATION_MS = 420 // --dur-long

interface Props {
  value: number | null
  format: (v: number | null) => string
  className?: string
}

const finite = (v: number | null): v is number => v !== null && Number.isFinite(v)

/**
 * A number that counts from its previous value to a new one. It shows its first value as is,
 * and under reduced motion it always renders the final value immediately.
 */
export function AnimatedNumber({ value, format, className }: Props) {
  const { reduced } = useMotion()
  const [prev, setPrev] = useState(value)
  const [from, setFrom] = useState<number | null>(null)
  const [frame, setFrame] = useState<number | null>(null)

  // A new value: start the tween from whatever is painted now (React's "adjust state on prop
  // change" pattern). Without motion, or with nothing to count between, it is shown at once.
  if (value !== prev) {
    const painted = frame ?? from ?? prev
    const animate = !reduced && typeof requestAnimationFrame === 'function' && finite(painted) && finite(value)
    setPrev(value)
    setFrom(animate ? painted : null)
    setFrame(null)
  }

  useLayoutEffect(() => {
    if (from === null || !finite(value)) return
    const start = performance.now()
    let raf = requestAnimationFrame(function step(now) {
      const t = Math.min(1, Math.max(0, (now - start) / DURATION_MS))
      if (t < 1) {
        setFrame(from + (value - from) * (1 - (1 - t) ** 3))
        raf = requestAnimationFrame(step)
      } else {
        setFrame(null)
        setFrom(null)
      }
    })
    return () => cancelAnimationFrame(raf)
  }, [from, value])

  // The accessible text is always the final value; only the painted digits move.
  return (
    <span className={className} data-value={value ?? ''}>
      <span aria-hidden="true">{format(frame ?? from ?? value)}</span>
      <span className="visually-hidden">{format(value)}</span>
    </span>
  )
}
