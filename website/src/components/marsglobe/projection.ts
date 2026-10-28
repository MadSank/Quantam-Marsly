// Projection maths for the Mars globe view. The three.js sphere and the SVG overlay (pins,
// callouts, footprints) share these functions, so markers sit exactly on the rendered planet.
// The mesh rotation is Euler XYZ (pitch about x, yaw about y): v' = Rx(pitch) · Ry(yaw) · v,
// matching three.js; the orthographic camera looks down −z, so screen x = x and screen y = −y.
import { rotationForLongitude, surfacePoint } from '../globe/spec'

export interface GlobeView {
  /** Rotation about the planet's polar axis, radians. */
  yaw: number
  /** Tilt about the screen's horizontal axis, radians. */
  pitch: number
  /** Orthographic camera zoom; 1 shows the whole planet. */
  zoom: number
}

export interface Size {
  width: number
  height: number
}

export interface Projected {
  x: number
  y: number
  /** Depth toward the viewer, −1 … 1; ≤ 0 is the far side. */
  z: number
  visible: boolean
}

export const MIN_ZOOM = 1
export const MAX_ZOOM = 800
export const MAX_PITCH = (85 * Math.PI) / 180

/** Planet radius in pixels at zoom 1. */
export function baseRadius(size: Size): number {
  return Math.min(size.width, size.height) * 0.36
}

export function rotate([x, y, z]: [number, number, number], view: Pick<GlobeView, 'yaw' | 'pitch'>): [number, number, number] {
  const cy = Math.cos(view.yaw)
  const sy = Math.sin(view.yaw)
  const x1 = x * cy + z * sy
  const z1 = -x * sy + z * cy
  const cp = Math.cos(view.pitch)
  const sp = Math.sin(view.pitch)
  return [x1, y * cp - z1 * sp, y * sp + z1 * cp]
}

/** Where (lat, lon) lands on screen for an orthographic globe centred in `size`. */
export function projectOrtho(lat: number, lonE: number, view: GlobeView, size: Size): Projected {
  const [x, y, z] = rotate(surfacePoint(lat, lonE), view)
  const r = baseRadius(size) * view.zoom
  return { x: size.width / 2 + x * r, y: size.height / 2 - y * r, z, visible: z > 0 }
}

/** The view that puts (lat, lon) at the centre of the screen. */
export function viewFor(lat: number, lonE: number, zoom: number): GlobeView {
  return { yaw: rotationForLongitude(lonE), pitch: (lat * Math.PI) / 180, zoom }
}

/** `to` shifted by whole turns to sit nearest `from`, so a fly-to takes the short way round. */
export function nearestAngle(from: number, to: number): number {
  const turn = 2 * Math.PI
  return to + Math.round((from - to) / turn) * turn
}

export function clampView(v: GlobeView): GlobeView {
  return {
    yaw: v.yaw,
    pitch: Math.max(-MAX_PITCH, Math.min(MAX_PITCH, v.pitch)),
    zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, v.zoom)),
  }
}

/** Longitude east in 0…360 to −180…180, the layout of the colour texture. */
export function signedLongitude(lonE: number): number {
  const l = ((lonE % 360) + 360) % 360
  return l > 180 ? l - 360 : l
}

/** Flat equirectangular map (the no-WebGL fallback): −180…180 left to right, 90…−90 top to bottom. */
export function projectEquirect(lat: number, lonE: number, rect: { x: number; y: number } & Size): Projected {
  return {
    x: rect.x + ((signedLongitude(lonE) + 180) / 360) * rect.width,
    y: rect.y + ((90 - lat) / 180) * rect.height,
    z: 1,
    visible: true,
  }
}

export interface Insets {
  top: number
  right: number
  bottom: number
  left: number
}

/** Room the flat map leaves for the site list (left) and the site card (right). */
export const FLAT_INSETS: Insets = { top: 24, right: 388, bottom: 40, left: 320 }

/** The flat map's rectangle: 2:1, centred in the stage area left free by `insets`. */
export function flatMapRect(size: Size, insets: Insets = FLAT_INSETS): { x: number; y: number } & Size {
  const freeW = Math.max(0, size.width - insets.left - insets.right)
  const freeH = Math.max(0, size.height - insets.top - insets.bottom)
  const width = Math.min(freeW, freeH * 2)
  const height = width / 2
  return { x: insets.left + (freeW - width) / 2, y: insets.top + (freeH - height) / 2, width, height }
}

/**
 * Zoom that frames a site's strip at about 40% of the stage height: 1.2 / (angular extent in
 * radians), since the planet's radius is 0.36 of the stage's shorter side.
 */
export function zoomForFootprint(footprint: Array<[number, number]>): number {
  const lats = footprint.map(([lat]) => lat)
  const lons = footprint.map(([, lon]) => lon)
  const midLat = ((Math.min(...lats) + Math.max(...lats)) / 2) * (Math.PI / 180)
  const dLat = Math.max(...lats) - Math.min(...lats)
  const dLon = (Math.max(...lons) - Math.min(...lons)) * Math.cos(midLat)
  const extent = (Math.max(dLat, dLon) * Math.PI) / 180
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, 1.2 / Math.max(extent, 1e-6)))
}
