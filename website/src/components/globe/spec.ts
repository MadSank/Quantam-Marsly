// Globe parameters exported by scripts/export_images.py (img/manifest.json "globe").

export interface GlobeSpec {
  url: string
  credit: string
  /** Longitude (degrees east) facing the viewer at rest. */
  front_longitude: number
  /** Direction to the light, in view space (x right, y up, z toward the viewer). */
  light: [number, number, number]
  ambient: number
  diffuse: number
  textures: Record<'1k' | '2k', { src: string; width: number; height: number }>
  still: { src: string; width: number; height: number }
}

/**
 * three.js SphereGeometry maps texture u = 0 to the −x axis, so longitude −90° faces the
 * camera by default. Rotating about y by −(lon + 90)° brings `lon` to the front.
 */
export function rotationForLongitude(longitude: number): number {
  return (-(longitude + 90) * Math.PI) / 180
}

export interface GlobeTarget {
  lat: number
  lon: number
}

/**
 * Mesh rotation (Euler XYZ: y applied first, then x) that turns (lat, lon) to face the
 * viewer: y brings the longitude to the front, then x tilts its latitude onto the equator.
 */
export function orientationFor(latitude: number, longitude: number): { x: number; y: number } {
  return { x: (latitude * Math.PI) / 180, y: rotationForLongitude(longitude) }
}

/** Unit vector toward (lat, lon) on the sphere, before rotation; for pins. */
export function surfacePoint(latitude: number, longitude: number): [number, number, number] {
  const lat = (latitude * Math.PI) / 180
  const phi = ((longitude + 180) * Math.PI) / 180
  return [-Math.cos(phi) * Math.cos(lat), Math.sin(lat), Math.sin(phi) * Math.cos(lat)]
}
