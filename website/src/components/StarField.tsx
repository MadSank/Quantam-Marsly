import { useEffect, useRef } from 'react'

// A sparse, static star field: magnitudes follow a steep power law (most stars faint),
// with a slight spread of colour temperature. No twinkle, no glow, no nebula.

const STARS_PER_MEGAPIXEL = 260
const SEED = 20261028

function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function tokens(el: HTMLElement): string[] {
  const css = getComputedStyle(el)
  return ['--star-cool', '--star-neutral', '--star-neutral', '--star-neutral', '--star-warm'].map((t) =>
    css.getPropertyValue(t).trim(),
  )
}

export function StarField({ className }: { className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return

    const draw = () => {
      const { width, height } = el.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      el.width = Math.round(width * dpr)
      el.height = Math.round(height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)
      const colours = tokens(el)
      const rand = mulberry32(SEED)
      // Draw over a fixed 2560 x 1600 field so stars keep their places as the window resizes.
      const count = Math.round((2560 * 1600 * STARS_PER_MEGAPIXEL) / 1e6)
      for (let i = 0; i < count; i++) {
        const x = rand() * 2560
        const y = rand() * 1600
        const m = rand() ** 3.2 // 0 faint ... 1 bright
        const colour = colours[Math.floor(rand() * colours.length)]
        if (x > width || y > height) continue
        ctx.globalAlpha = 0.18 + 0.82 * m
        ctx.fillStyle = colour
        ctx.beginPath()
        ctx.arc(x, y, 0.35 + 1.05 * m, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return <canvas ref={canvas} className={className} aria-hidden="true" />
}
