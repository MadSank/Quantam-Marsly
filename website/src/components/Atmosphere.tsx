import type { MarsContoursFile } from '../data/types'
import { useJson } from '../lib/useJson'

/**
 * The page ground behind every explorer panel: contour lines traced from the current
 * site's HiRISE DTM (scripts/export_contours.py), a vignette and a fine grain. Purely
 * decorative: hidden from assistive technology and from pointer input.
 */
export function Atmosphere({ siteId }: { siteId: string }) {
  const contours = useJson<MarsContoursFile>(`mars/contours/${siteId}.json`)

  return (
    <div className="atmosphere" aria-hidden="true">
      {contours && (
        <svg
          key={contours.site_id}
          className="contours"
          viewBox={`0 0 ${contours.width} ${contours.height}`}
          preserveAspectRatio="xMidYMid slice"
          data-site={contours.site_id}
        >
          {contours.levels.map((level) => (
            <path key={level.elevation_m} d={level.d} className={level.index ? 'contour contour--index' : 'contour'} />
          ))}
        </svg>
      )}
      <div className="vignette" />
      <div className="grain" />
    </div>
  )
}
