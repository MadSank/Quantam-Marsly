import type { MarsSite } from '../../data/types'

export function latLabel(lat: number) {
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}`
}

export function lonLabel(lon: number) {
  return `${lon.toFixed(2)}° E`
}

/** One factual line from sites.json (no claims beyond the exported metadata). */
export function siteDescription(s: MarsSite): string {
  return `HiRISE orthoimage and DTM at ${s.native_resolution_m.toFixed(2)} m per pixel, held out as the test site of fold ${s.fold}.`
}
