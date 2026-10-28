// The Mars globe view's planet: a textured sphere with MOLA relief, drawn for a view the page
// controls (yaw, pitch, zoom). Pins and callouts are an SVG overlay that uses the same maths
// (projection.ts), so this component renders the planet only. Loaded lazily with three.js.
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { useLayoutEffect, useRef } from 'react'
import { LinearMipmapLinearFilter, NoColorSpace, SRGBColorSpace, TextureLoader, Vector2, type Mesh } from 'three'
import { baseRadius, type GlobeView, type Size } from './projection'

export interface GlobeSphereProps {
  textureUrl: string
  normalUrl: string
  view: GlobeView
  size: Size
  onReady?: () => void
}

// Light from the upper left, fixed to the viewer, so relief reads the same wherever the planet turns.
const LIGHT: [number, number, number] = [-0.55, 0.5, 0.67]
const AMBIENT = 0.14
const DIFFUSE = 1.05
const RELIEF = new Vector2(1, 1)

function Planet({ textureUrl, normalUrl, view, size, onReady }: GlobeSphereProps) {
  const [colour, normal] = useLoader(TextureLoader, [textureUrl, normalUrl])
  const mesh = useRef<Mesh>(null)
  const get = useThree((s) => s.get)
  const invalidate = useThree((s) => s.invalidate)
  const announced = useRef(false)

  useLayoutEffect(() => {
    if (!mesh.current) return
    mesh.current.rotation.set(view.pitch, view.yaw, 0)
    const { camera } = get()
    camera.zoom = view.zoom
    camera.updateProjectionMatrix()
    invalidate()
    if (!announced.current) {
      announced.current = true
      onReady?.()
    }
  }, [view, get, invalidate, onReady])

  return (
    <mesh ref={mesh} scale={baseRadius(size)}>
      <sphereGeometry args={[1, 192, 96]} />
      <meshLambertMaterial normalScale={RELIEF}>
        <primitive attach="map" object={colour} colorSpace={SRGBColorSpace} minFilter={LinearMipmapLinearFilter} anisotropy={8} />
        <primitive attach="normalMap" object={normal} colorSpace={NoColorSpace} minFilter={LinearMipmapLinearFilter} anisotropy={8} />
      </meshLambertMaterial>
    </mesh>
  )
}

export default function GlobeSphere(props: GlobeSphereProps) {
  return (
    <Canvas
      flat
      orthographic
      dpr={[1, 2]}
      frameloop="demand"
      camera={{ position: [0, 0, 4096], near: 1, far: 8192, zoom: props.view.zoom }}
      gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
    >
      {/* three's Lambert divides by π, so intensities carry π (as in the landing globe). */}
      <ambientLight intensity={AMBIENT * Math.PI} />
      <directionalLight position={LIGHT} intensity={DIFFUSE * Math.PI} />
      <Planet {...props} />
    </Canvas>
  )
}
