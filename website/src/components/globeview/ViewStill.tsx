import { dataUrl } from '../../lib/dataUrl'
import { baseRadius, type Size } from './projection'

interface Props {
  still: { src: string; width: number; height: number }
  size: Size
  zoom: number
}

/**
 * The globe page's opening view, pre-rendered (export_images.py render_view_still) with the live
 * sphere's view and light. It paints before three.js and the textures arrive, sits exactly where
 * the sphere will be drawn, and fades out once the sphere is ready (globe.css, [data-ready]).
 * Away from the opening zoom it is not shown, since it would no longer match.
 */
export function ViewStill({ still, size, zoom }: Props) {
  if (zoom > 1.05) return null
  const r = baseRadius(size) * zoom
  return (
    <img
      className="globe-view-still"
      src={dataUrl(still.src)}
      width={still.width}
      height={still.height}
      alt=""
      draggable={false}
      fetchPriority="high"
      style={{ left: size.width / 2 - r, top: size.height / 2 - r, width: 2 * r, height: 2 * r }}
    />
  )
}
