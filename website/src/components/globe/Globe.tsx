import { Component, lazy, Suspense, useState, type ReactNode } from 'react'
import { useMotion } from '../motionContext'
import type { GlobeSpec, GlobeTarget } from './spec'

const GlobeCanvas = lazy(() => import('./GlobeCanvas'))

const dataUrl = (rel: string) => `${import.meta.env.BASE_URL}data/${rel}`

// One revolution about every three minutes.
export const SLOW_ROTATION = (2 * Math.PI) / 180

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

/** Keeps the still in place if the WebGL scene fails to load or render. */
class StayStill extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? null : this.props.children
  }
}

export interface GlobeProps {
  spec: GlobeSpec
  className?: string
  /** Texture for the live sphere; '2k' for large globes. */
  resolution?: '1k' | '2k'
  rotationSpeed?: number
  /**
   * Locator mode: the globe holds this point toward the viewer with a pin and eases to
   * each new target. It is state, not ambient motion, so it renders under reduced motion
   * too, jumping instead of easing. Without WebGL a locator renders nothing.
   */
  target?: GlobeTarget
  /** CSS custom property holding the pin colour, read from the canvas's own element. */
  pinToken?: string
  /** Pins and overlays for the live scene (Phase 2). */
  children?: ReactNode
}

/**
 * A planet: the pre-rendered still paints first; when motion is allowed and WebGL is
 * available, the three.js sphere loads lazily and fades in over it. Under reduced motion
 * only the still is shown and three.js is never downloaded.
 */
export function Globe({ spec, className, resolution = '2k', rotationSpeed = SLOW_ROTATION, target, pinToken, children }: GlobeProps) {
  const { allowed, reduced } = useMotion()
  const [live] = useState(() => (target !== undefined || !reduced) && webglAvailable())
  const [ready, setReady] = useState(false)
  if (target && !live) return null

  return (
    <div className={['globe', ready ? 'is-live' : '', className].filter(Boolean).join(' ')} aria-hidden="true">
      {!target && <img className="globe-still" src={dataUrl(spec.still.src)} width={spec.still.width} height={spec.still.height} alt="" />}
      {live && (
        <StayStill>
          <Suspense fallback={null}>
            <div className="globe-live">
              <GlobeCanvas
                spec={spec}
                textureUrl={dataUrl(spec.textures[resolution].src)}
                rotationSpeed={rotationSpeed}
                running={allowed}
                onReady={() => setReady(true)}
                target={target}
                animate={!reduced}
                pinToken={pinToken}
              >
                {children}
              </GlobeCanvas>
            </div>
          </Suspense>
        </StayStill>
      )}
    </div>
  )
}
