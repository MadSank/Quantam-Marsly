// Canvas drawing for the Mars tile overlay, in viewport pixels. Tiles are positioned in native
// ortho pixels (row, col, size_native) and mapped through the pan/zoom view.
import type { MarsTilesFile } from '../../data/types'
import type { Size, View } from '../panzoom/math'
import { outcome, predictionStyle, truthStyle, type ModelKey, type Outcome } from './style'

export type { ModelKey }

export interface OverlayColours {
  model: string
  truthSafe: string
  truthUnsafe: string
  focus: string
  ink: string
}

export interface DrawOptions {
  tiles: MarsTilesFile
  model: ModelKey
  seed: string
  view: View
  viewport: Size
  colours: OverlayColours
  showTruth: boolean
  selected: number | null
}

const FILL_ALPHA = 0.3
const HATCH_SPACING = 6 // screen pixels between hatch lines
const DASH = [4, 3]

export function countOutcomes(tiles: MarsTilesFile, model: ModelKey, seed: string): Record<Outcome, number> {
  const counts: Record<Outcome, number> = { 'safe-agree': 0, 'safe-wrong': 0, 'unsafe-wrong': 0, 'unsafe-agree': 0 }
  const pred = tiles.pred[model][seed].safe
  tiles.true.safe.forEach((t, i) => counts[outcome(pred[i], t)]++)
  return counts
}

const indexCache = new WeakMap<MarsTilesFile, Map<string, number>>()

/** The tile covering a content point (native pixels), or null. */
export function tileAt(tiles: MarsTilesFile, x: number, y: number): number | null {
  let index = indexCache.get(tiles)
  if (!index) {
    index = new Map(tiles.row.map((r, i) => [`${r}:${tiles.col[i]}`, i]))
    indexCache.set(tiles, index)
  }
  const s = tiles.size_native
  return index.get(`${Math.floor(y / s) * s}:${Math.floor(x / s) * s}`) ?? null
}

function cornerPath(p: Path2D, x: number, y: number, size: number) {
  p.moveTo(x, y)
  p.lineTo(x + size, y)
  p.lineTo(x, y + size)
  p.closePath()
}

function rectPath(p: Path2D, x: number, y: number, s: number, inset: number) {
  p.rect(x + inset, y + inset, Math.max(0.5, s - 2 * inset), Math.max(0.5, s - 2 * inset))
}

export function drawOverlay(ctx: CanvasRenderingContext2D, o: DrawOptions) {
  const { tiles, view, viewport, colours } = o
  const pred = tiles.pred[o.model][o.seed]?.safe
  ctx.clearRect(0, 0, viewport.width, viewport.height)
  if (!pred) return

  const s = tiles.size_native * view.scale
  // Inset outlines once tiles are big enough, so neighbours keep separate edges.
  const inset = s >= 8 ? 1 : 0
  const fills = new Path2D()
  const hatched = new Path2D()
  const solid = new Path2D()
  const dashed = new Path2D()
  const truthSafe = new Path2D()
  const truthUnsafe = new Path2D()

  for (let i = 0; i < pred.length; i++) {
    const x = tiles.col[i] * view.scale + view.x
    const y = tiles.row[i] * view.scale + view.y
    if (x + s < 0 || y + s < 0 || x > viewport.width || y > viewport.height) continue
    const style = predictionStyle(outcome(pred[i], tiles.true.safe[i]))
    if (style.fill) rectPath(fills, x, y, s, inset)
    if (style.hatch) rectPath(hatched, x, y, s, inset)
    if (style.outline === 'solid') rectPath(solid, x, y, s, inset + 0.5)
    if (style.outline === 'dashed') rectPath(dashed, x, y, s, inset + 0.5)
    if (o.showTruth) {
      const t = truthStyle(tiles.true.safe[i])
      const corner = Math.max(2, Math.min(16, s * 0.3))
      cornerPath(t.colour === 'truth-safe' ? truthSafe : truthUnsafe, x + inset, y + inset, corner)
    }
  }

  ctx.save()
  ctx.fillStyle = colours.model
  ctx.globalAlpha = FILL_ALPHA
  ctx.fill(fills)
  ctx.globalAlpha = 1

  // Hatching: diagonal lines across the viewport, clipped to the hatched tiles. Dark ink on the
  // model-colour fill, so the pattern reads by value as well as by shape.
  if (s >= 6) {
    ctx.save()
    ctx.clip(hatched)
    ctx.strokeStyle = colours.ink
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let k = -viewport.height; k < viewport.width; k += HATCH_SPACING) {
      ctx.moveTo(k, viewport.height)
      ctx.lineTo(k + viewport.height, 0)
    }
    ctx.stroke()
    ctx.restore()
  }

  ctx.strokeStyle = colours.model
  ctx.lineWidth = s >= 8 ? 1.5 : 1
  ctx.stroke(solid)
  ctx.setLineDash(DASH)
  ctx.stroke(dashed)
  ctx.setLineDash([])

  if (o.showTruth) {
    ctx.fillStyle = colours.truthSafe
    ctx.fill(truthSafe)
    ctx.fillStyle = colours.truthUnsafe
    ctx.fill(truthUnsafe)
  }

  // The inspected tile: a focus-colour ring on a dark halo, so it shows over any encoding.
  if (o.selected !== null && o.selected < pred.length) {
    const x = tiles.col[o.selected] * view.scale + view.x
    const y = tiles.row[o.selected] * view.scale + view.y
    const w = Math.max(6, s + 6)
    ctx.strokeStyle = colours.ink
    ctx.lineWidth = 5
    ctx.strokeRect(x - 3, y - 3, w, w)
    ctx.strokeStyle = colours.focus
    ctx.lineWidth = 2.5
    ctx.strokeRect(x - 3, y - 3, w, w)
  }
  ctx.restore()
}
