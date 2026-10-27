// The live globe: a textured, Lambert-lit sphere in three.js. Loaded lazily by <Globe>.
// Lighting matches scripts/export_images.py render_still(), so the cross-fade from the
// pre-rendered still is seamless. Reused by the Phase 2 globe (pins go in `children`).
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { LinearMipmapLinearFilter, SRGBColorSpace, TextureLoader, type Mesh } from 'three'
import { rotationForLongitude, type GlobeSpec } from './spec'

export interface GlobeCanvasProps {
  spec: GlobeSpec
  textureUrl: string
  /** Radians per second; 0 holds still. */
  rotationSpeed: number
  running: boolean
  onReady?: () => void
  children?: ReactNode
}

function Sphere({ spec, textureUrl, rotationSpeed, running, onReady, children }: GlobeCanvasProps) {
  const texture = useLoader(TextureLoader, textureUrl)
  const mesh = useRef<Mesh>(null)
  // Orthographic camera at zoom 1 (one unit per pixel): scale the unit sphere to fill the square canvas.
  const radius = useThree((state) => Math.min(state.size.width, state.size.height) / 2)
  const announced = useRef(false)

  useLayoutEffect(() => {
    if (mesh.current) mesh.current.rotation.y = rotationForLongitude(spec.front_longitude)
  }, [spec.front_longitude])

  useFrame((_, delta) => {
    if (running && mesh.current) mesh.current.rotation.y += rotationSpeed * Math.min(delta, 0.1)
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
      {children}
    </mesh>
  )
}

export default function GlobeCanvas(props: GlobeCanvasProps) {
  const { spec, running } = props
  const [lx, ly, lz] = spec.light

  return (
    <Canvas
      flat
      orthographic
      dpr={[1, 2]}
      // While paused, render on demand only: the last frame stays on screen.
      frameloop={running ? 'always' : 'demand'}
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
