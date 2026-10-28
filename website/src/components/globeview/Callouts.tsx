import type { MarsSite } from '../../data/types'
import { dataUrl } from '../../lib/dataUrl'
import type { Projected } from './projection'

/** Callout direction per site (screen pixels from the pin), chosen so neighbouring labels clear each other. */
const OFFSETS: Record<string, [number, number]> = {
  site_01: [70, -64],
  site_02: [-78, 56],
  site_03: [-86, 46],
  site_04: [-80, -58],
  site_05: [-74, -70],
}
// On the flat map, site 01 sits right of the 03/05 cluster, near the card: lift its label above them.
const FLAT_OFFSETS: Record<string, [number, number]> = { ...OFFSETS, site_01: [-12, -112] }
const DEFAULT_OFFSET: [number, number] = [70, -60]
const LABEL_H = 30 // label box height, px
const labelWidth = (name: string) => 44 + name.length * 8.2 // index, padding and name at --text-sm

export interface Rect {
  left: number
  top: number
  right: number
  bottom: number
}

const hits = (a: Rect, b: Rect) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom

/** The label's box for a pin at (x, y) and offset dx, dy. */
function labelRect(x: number, y: number, dx: number, dy: number, name: string): Rect {
  const w = labelWidth(name)
  const anchor = x + dx + Math.sign(dx) * ELBOW
  return { left: dx < 0 ? anchor - w : anchor, right: dx < 0 ? anchor : anchor + w, top: y + dy - LABEL_H / 2, bottom: y + dy + LABEL_H / 2 }
}
const ELBOW = 22 // horizontal run of the leader line before the label

/**
 * The site's own HiRISE preview, placed on the planet by its geographic corners. Close to a
 * site the projection is locally affine, so three projected corners define the CSS matrix.
 */
function Drape({ site, project }: { site: MarsSite; project: Props['project'] }) {
  if (!site.preview) return null
  const [tl, tr, bl] = (['top_left', 'top_right', 'bottom_left'] as const).map((k) => project(...site.image_corners[k]))
  if (![tl, tr, bl].every((p) => p.visible)) return null
  const { width: w, height: h } = site.preview
  const m = [(tr.x - tl.x) / w, (tr.y - tl.y) / w, (bl.x - tl.x) / h, (bl.y - tl.y) / h, tl.x, tl.y]
  return (
    <img
      className="drape"
      data-site={site.site_id}
      src={dataUrl(site.preview.src)}
      width={w}
      height={h}
      alt=""
      draggable={false}
      style={{ transform: `matrix(${m.map((v) => v.toFixed(6)).join(',')})` }}
    />
  )
}

interface Props {
  sites: MarsSite[]
  project: (lat: number, lonE: number) => Projected
  selected: string | null
  /** Draw strip footprints (close zoom on the globe). */
  footprints: boolean
  /** Callout layout: the globe's, or the flat map's (labels kept clear of the card). */
  layout?: 'globe' | 'flat'
  /** Areas labels must stay out of (the site list, the open card), in stage pixels. */
  avoid?: Rect[]
  onSelect: (siteId: string) => void
}

/**
 * Pins at each site's exact centre with a leader line and a label box, plus strip footprints
 * and the selected site's draped preview at close zoom. Far-side pins and their labels are
 * hidden. Pointer only: the site list beside the globe is the keyboard and screen-reader route
 * to the same actions.
 */
export function Callouts({ sites, project, selected, footprints, layout = 'globe', avoid = [], onSelect }: Props) {
  const table = layout === 'flat' ? FLAT_OFFSETS : OFFSETS
  // Every visible pin, so a label never covers another site's pin.
  const pins = sites.flatMap((s) => {
    const p = project(s.center.lat, s.center.lon_e)
    return p.visible ? [{ id: s.site_id, rect: { left: p.x - 10, top: p.y - 10, right: p.x + 10, bottom: p.y + 10 } }] : []
  })
  // A label tries its preferred diagonal, then the mirrored ones, and takes the first that lands
  // in no panel and on no other pin; failing that, the least bad (a panel costs more than a pin).
  const offsetFor = (s: MarsSite, x: number, y: number): [number, number] => {
    const [dx, dy] = table[s.site_id] ?? DEFAULT_OFFSET
    const others = pins.filter((p) => p.id !== s.site_id).map((p) => p.rect)
    const cost = ([ox, oy]: [number, number]) => {
      const r = labelRect(x, y, ox, oy, s.name)
      return 3 * avoid.filter((a) => hits(r, a)).length + others.filter((o) => hits(r, o)).length
    }
    const options: Array<[number, number]> = [[dx, dy], [-dx, dy], [dx, -dy], [-dx, -dy]]
    return options.reduce((best, o) => (cost(o) < cost(best) ? o : best))
  }
  const drape = footprints && selected ? sites.find((s) => s.site_id === selected) : undefined
  return (
    <div className="callouts" aria-hidden="true">
      {drape && <Drape site={drape} project={project} />}
      <svg className="callout-lines">
        {footprints &&
          sites.map((s) => {
            const pts = s.footprint.map(([lat, lon]) => project(lat, lon))
            if (!pts.some((p) => p.visible)) return null
            return (
              <polygon
                key={`fp-${s.site_id}`}
                className="footprint"
                data-site={s.site_id}
                data-selected={s.site_id === selected || undefined}
                points={pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
              />
            )
          })}
        {sites.map((s) => {
          const p = project(s.center.lat, s.center.lon_e)
          if (!p.visible) return null
          const [dx, dy] = offsetFor(s, p.x, p.y)
          const ex = p.x + dx
          const ey = p.y + dy
          const end = ex + Math.sign(dx) * ELBOW
          return (
            <g key={s.site_id} data-site={s.site_id} data-selected={s.site_id === selected || undefined} className="callout-line">
              <polyline points={`${p.x},${p.y} ${ex},${ey} ${end},${ey}`} />
              <circle className="pin-ring" cx={p.x} cy={p.y} r={9} />
              <circle className="pin" cx={p.x} cy={p.y} r={4} />
            </g>
          )
        })}
      </svg>
      {sites.map((s, i) => {
        const p = project(s.center.lat, s.center.lon_e)
        if (!p.visible) return null
        const [dx, dy] = offsetFor(s, p.x, p.y)
        const x = p.x + dx + Math.sign(dx) * ELBOW
        const y = p.y + dy
        return (
          <button
            key={s.site_id}
            type="button"
            tabIndex={-1}
            className="callout"
            data-site={s.site_id}
            data-side={dx < 0 ? 'left' : 'right'}
            data-selected={s.site_id === selected || undefined}
            data-lat={s.center.lat}
            data-lon={s.center.lon_e}
            data-x={p.x.toFixed(2)}
            data-y={p.y.toFixed(2)}
            style={{ transform: `translate(${x}px, ${y}px)` }}
            onClick={() => onSelect(s.site_id)}
          >
            <span className="callout-box">
              <span className="callout-index num">{String(i + 1).padStart(2, '0')}</span>
              <span className="callout-name">{s.name}</span>
            </span>
          </button>
        )
      })}
      {/* Larger hit targets on the pins themselves. */}
      {sites.map((s) => {
        const p = project(s.center.lat, s.center.lon_e)
        if (!p.visible) return null
        return (
          <button
            key={`hit-${s.site_id}`}
            type="button"
            tabIndex={-1}
            className="pin-hit"
            data-site={s.site_id}
            style={{ transform: `translate(${p.x}px, ${p.y}px)` }}
            onClick={() => onSelect(s.site_id)}
          />
        )
      })}
    </div>
  )
}
