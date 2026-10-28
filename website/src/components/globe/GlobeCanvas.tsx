// The live globe: a textured, Lambert-lit sphere in three.js. Loaded lazily by <Globe>.
// Lighting matches scripts/export_images.py render_still(), so the cross-fade from the
// pre-rendered still is seamless. Reused by the Phase 2 globe (pins go in `children`).
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef, type ReactNode } from 'react'
import { LinearMipmapLinearFilter, SRGBColorSpace, TextureLoader, type Mesh } from 'three'
import { orientationFor, rotationForLongitude, surfacePoint, type GlobeSpec, type GlobeTarget } from './spec'

export interface GlobeCanvasProps {
  spec: GlobeSpec
  textureUrl: string
  /** Radians per second; 0 holds still. */
  rotationSpeed: number
  running: boolean
  onReady?: () => void
  /** Locator mode: hold `target` facing the viewer, with a pin on it, instead of spinning. */
  target?: GlobeTarget
  /** Ease to a new target; false jumps (reduced motion). */
  animate?: boolean
  /** CSS custom property for the pin colour; three.js materials cannot read CSS variables. */
  pinToken?: string
  children?: ReactNode
}

// Locator easing: the share of the remaining angle closed per second (exponential approach).
const EASE_RATE = 5
const SETTLED = 1e-4

function shortest(from: number, to: number): number {
  const d = (to - from) % (2 * Math.PI)
  return d > Math.PI ? d - 2 * Math.PI : d < -Math.PI ? d + 2 * Math.PI : d
}

function Sphere({ spec, textureUrl, rotationSpeed, running, onReady, target, animate = true, pinToken, children }: GlobeCanvasProps) {
  const texture = useLoader(TextureLoader, textureUrl)
  const mesh = useRef<Mesh>(null)
  // Orthographic camera at zoom 1 (one unit per pixel): scale the unit sphere to fill the square canvas.
  const radius = useThree((state) => Math.min(state.size.width, state.size.height) / 2)
  const invalidate = useThree((state) => state.invalidate)
  const canvas = useThree((state) => state.gl.domElement)
  const pinColor = useMemo(
    () => (pinToken ? getComputedStyle(canvas).getPropertyValue(pinToken).trim() || undefined : undefined),
    [canvas, pinToken],
  )
  const announced = useRef(false)
  const placed = useRef(false)

  useLayoutEffect(() => {
    if (!mesh.current) return
    if (target) {
      // First placement, or a jump under reduced motion: no easing.
      if (!placed.current || !animate) {
        const o = orientationFor(target.lat, target.lon)
        mesh.current.rotation.set(o.x, o.y, 0)
        placed.current = true
      }
      invalidate()
    } else mesh.current.rotation.y = rotationForLongitude(spec.front_longitude)
  }, [spec.front_longitude, target, animate, invalidate])

  useFrame((_, delta) => {
    if (mesh.current && target) {
      const o = orientationFor(target.lat, target.lon)
      const r = mesh.current.rotation
      const dy = shortest(r.y, o.y)
      const dx = o.x - r.x
      if (Math.abs(dx) > SETTLED || Math.abs(dy) > SETTLED) {
        const k = animate ? 1 - Math.exp(-EASE_RATE * Math.min(delta, 0.1)) : 1
        r.x += dx * k
        r.y += dy * k
        invalidate()
      }
    } else if (running && mesh.current) mesh.current.rotation.y += rotationSpeed * Math.min(delta, 0.1)
    if (!announced.current) {
      announced.current = true
      onReady?.()
    }
  })

  return (
    <mesh ref={mesh} scale={radius}>
      <sphereGeometry args={[1, 128, 64]} />
      <meshLambertMaterial>
        <primitive
          attach="map"
          object={texture}
          colorSpace={SRGBColorSpace}
          minFilter={LinearMipmapLinearFilter}
          anisotropy={4}
        />
      </meshLambertMaterial>
      {target && (
        <mesh position={surfacePoint(target.lat, target.lon)} scale={0.06}>
          <sphereGeometry args={[1, 16, 12]} />
          <meshBasicMaterial color={pinColor} />
        </mesh>
      )}
      {children}
    </mesh>
  )
}

export default function GlobeCanvas(props: GlobeCanvasProps) {
  const { spec, running, target } = props
  const [lx, ly, lz] = spec.light

  return (
    <Canvas
      flat
      orthographic
      dpr={[1, 2]}
      // While paused, render on demand only: the last frame stays on screen.
      // A locator renders on demand too: only while it eases to a new site.
      frameloop={running && !target ? 'always' : 'demand'}
      camera={{ position: [0, 0, 4096], near: 1, far: 8192 }}
      gl={{ alpha: true, antialias: true, powerPreference: 'low-power' }}
    >
      {/* three's Lambert divides by π, so intensities carry π to match the still's model. */}
      <ambientLight intensity={spec.ambient * Math.PI} />
      <directionalLight position={[lx, ly, lz]} intensity={spec.diffuse * Math.PI} />
      <Sphere {...props} />
    </Canvas>
  )
}
