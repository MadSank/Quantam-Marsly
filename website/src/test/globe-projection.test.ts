import { describe, expect, it } from 'vitest'
import { Euler, Matrix4, Vector3 } from 'three'
import sitesFile from '../../public/data/mars/sites.json'
import { surfacePoint } from '../components/globe/spec'
import { nearestAngle, projectEquirect, projectOrtho, rotate, signedLongitude, viewFor } from '../components/globeview/projection'
import type { MarsSitesFile } from '../data/types'

const sites = (sitesFile as MarsSitesFile).sites
const size = { width: 1600, height: 900 }

describe('globe projection', () => {
  it('rotates exactly as three.js does for the mesh (Euler XYZ)', () => {
    for (const [yaw, pitch] of [[0.3, -0.7], [-2.1, 1.2], [4, 0.05]]) {
      const p = surfacePoint(23, 140)
      const ours = rotate(p, { yaw, pitch })
      const m = new Matrix4().makeRotationFromEuler(new Euler(pitch, yaw, 0, 'XYZ'))
      const theirs = new Vector3(...p).applyMatrix4(m)
      expect(ours[0]).toBeCloseTo(theirs.x, 10)
      expect(ours[1]).toBeCloseTo(theirs.y, 10)
      expect(ours[2]).toBeCloseTo(theirs.z, 10)
    }
  })

  it.each(sites.map((s) => [s.site_id, s.center.lat, s.center.lon_e] as const))(
    '%s: the view for its centre puts its pin at the screen centre, facing the viewer',
    (_, lat, lon) => {
      const p = projectOrtho(lat, lon, viewFor(lat, lon, 7), size)
      expect(p.x).toBeCloseTo(size.width / 2, 6)
      expect(p.y).toBeCloseTo(size.height / 2, 6)
      expect(p.z).toBeCloseTo(1, 6)
      expect(p.visible).toBe(true)
    },
  )

  it('puts north up and east to the right', () => {
    const view = viewFor(0, 0, 1)
    const centre = projectOrtho(0, 0, view, size)
    expect(projectOrtho(10, 0, view, size).y).toBeLessThan(centre.y)
    expect(projectOrtho(0, 10, view, size).x).toBeGreaterThan(centre.x)
  })

  it('hides the far side', () => {
    const view = viewFor(0, 0, 1)
    expect(projectOrtho(0, 180, view, size).visible).toBe(false)
    expect(projectOrtho(0, 80, view, size).visible).toBe(true)
  })

  it('takes the short way round', () => {
    expect(nearestAngle(0.1, 2 * Math.PI - 0.1)).toBeCloseTo(-0.1)
    expect(nearestAngle(10, 0.5)).toBeCloseTo(0.5 + 2 * Math.PI * 2)
  })

  it('maps east longitudes onto the −180…180 flat map', () => {
    expect(signedLongitude(300.9691)).toBeCloseTo(-59.0309)
    const rect = { x: 0, y: 0, width: 360, height: 180 }
    const p = projectEquirect(-13.8592, 300.9691, rect)
    expect(p.x).toBeCloseTo(180 - 59.0309)
    expect(p.y).toBeCloseTo(90 + 13.8592)
  })
})
