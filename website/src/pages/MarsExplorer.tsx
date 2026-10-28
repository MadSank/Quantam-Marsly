import { useState } from 'react'
import { useSearchParams } from 'react-router'
import sitesFile from '../../public/data/mars/sites.json'
import imageManifest from '../../public/data/img/manifest.json'
import { niceScaleBar } from '../components/panzoom/math'
import { PanZoomViewport } from '../components/panzoom/PanZoomViewport'
import { Rail } from '../components/Rail'
import { WorldFooter, WorldHeader } from '../components/WorldChrome'
import type { MarsSite, MarsSitesFile } from '../data/types'
import { dataUrl } from '../lib/dataUrl'
import { WorldLayout } from '../theme/WorldLayout'
import '../styles/explorer.css'

const sites = (sitesFile as MarsSitesFile).sites
const stretch = imageManifest.mars as unknown as Record<string, { display_stretch: { percentiles: [number, number] } }>
const fmt = new Intl.NumberFormat('en-GB')

// Zoom stops at twice the preview's own pixel density, so a downsampled preview is never blown up further.
const OVERZOOM = 2

function latLabel(lat: number) {
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}`
}

function formatLength(m: number) {
  return m >= 1000 ? `${fmt.format(m / 1000)} km` : `${fmt.format(m)} m`
}

/** Read ?site= and fall back to the first site; other params (model, seed) are kept as they are. */
function useSite(): [MarsSite, URLSearchParams] {
  const [params] = useSearchParams()
  const site = sites.find((s) => s.site_id === params.get('site')) ?? sites[0]
  return [site, params]
}

function StripImage({ site }: { site: MarsSite }) {
  const [loaded, setLoaded] = useState(false)
  const size = { width: site.native_width_px, height: site.native_height_px }
  return (
    <>
      {site.thumb && <img className="strip-placeholder" src={dataUrl(site.thumb.src)} alt="" style={size} draggable={false} />}
      {site.preview && (
        <img
          className="strip-image"
          src={dataUrl(site.preview.src)}
          width={site.preview.width}
          height={site.preview.height}
          alt={`HiRISE orthoimage ${site.ortho_product_id}, ${site.name}`}
          style={size}
          draggable={false}
          decoding="async"
          data-loaded={loaded || undefined}
          onLoad={() => setLoaded(true)}
        />
      )}
    </>
  )
}

function SiteFacts({ site }: { site: MarsSite }) {
  const p = stretch[site.site_id]?.display_stretch.percentiles
  const rows: Array<[string, string]> = [
    ['Tiles', fmt.format(site.n_tiles)],
    ['Held out in', `fold ${site.fold}`],
    ['Centre', `${latLabel(site.center.lat)}, ${site.center.lon_e.toFixed(2)}° E`],
    ['Ortho pixel', `${site.native_resolution_m.toFixed(2)} m`],
    ['Strip size', `${fmt.format(site.native_width_px)} × ${fmt.format(site.native_height_px)} px`],
    ['Orthoimage', site.ortho_product_id],
    ['DTM', site.dtm_product_id],
  ]
  return (
    <aside className="facts" aria-labelledby="facts-title">
      <p className="facts-kicker">Test site {site.site_id.replace('site_0', '')} of {sites.length}</p>
      <h1 id="facts-title">{site.name}</h1>
      <dl className="facts-list">
        {rows.map(([k, v]) => (
          <div key={k} className={k === 'Orthoimage' || k === 'DTM' ? 'facts-row-id' : undefined}>
            <dt>{k}</dt>
            <dd className={k === 'Orthoimage' || k === 'DTM' ? 'facts-id' : 'num'}>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="facts-note">
        Each site is one leave-one-site-out fold: the models were trained on three other sites, validated on a fourth and
        tested on this one.
      </p>
      {p && (
        <p className="facts-note">
          Display stretch only ({p[0]}–{p[1]} percentile); the models saw the unstretched ortho tiles.
        </p>
      )}
    </aside>
  )
}

export function MarsExplorer() {
  const [site, params] = useSite()
  const to = (id: string) => {
    const next = new URLSearchParams(params)
    next.set('site', id)
    return { search: `?${next}` }
  }
  const maxScale = OVERZOOM * (site.preview?.scale_from_native.x ?? 1)

  return (
    <WorldLayout world="mars">
      <div className="explorer">
        <WorldHeader world="mars" />
        <main className="explorer-main">
          <Rail
            label="Sites"
            selected={site.site_id}
            to={to}
            items={sites.map((s) => ({
              id: s.site_id,
              label: s.name,
              detail: `${fmt.format(s.n_tiles)} tiles`,
              thumb: s.thumb && { ...s.thumb, alt: '' },
            }))}
          />
          <section className="image-panel" aria-label={`${site.name} strip`}>
            <PanZoomViewport
              key={site.site_id}
              content={{ width: site.native_width_px, height: site.native_height_px }}
              maxScale={maxScale}
              label={`${site.name} orthoimage strip`}
              hud={(view) => {
                const bar = niceScaleBar(view.scale, site.native_resolution_m)
                return (
                  <div className="scale-bar" aria-hidden="true">
                    <span className="scale-bar-line" style={{ width: bar.px }} />
                    <span className="num">{formatLength(bar.length)}</span>
                  </div>
                )
              }}
            >
              <StripImage site={site} />
            </PanZoomViewport>
          </section>
          <SiteFacts site={site} />
        </main>
        <WorldFooter />
      </div>
    </WorldLayout>
  )
}
