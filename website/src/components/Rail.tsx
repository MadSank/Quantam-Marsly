import type { KeyboardEvent } from 'react'
import { Link, type To } from 'react-router'
import type { ImageRef } from '../data/types'
import { dataUrl } from '../lib/dataUrl'

export interface RailItem {
  id: string
  label: string
  /** A short figure under the label, e.g. "2,195 tiles". */
  detail: string
  thumb?: (ImageRef & { alt: string }) | null
}

interface Props {
  label: string
  items: RailItem[]
  selected: string
  /** Where each item links; selection lives in the URL so views can be shared. */
  to: (id: string) => To
}

/** Shared item rail (Mars sites, Moon classes). Up/Down/Home/End move between items. */
export function Rail({ label, items, selected, to }: Props) {
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    const links = [...e.currentTarget.querySelectorAll<HTMLAnchorElement>('a.rail-item')]
    const i = links.indexOf(document.activeElement as HTMLAnchorElement)
    if (i < 0) return
    const next =
      e.key === 'ArrowDown' || e.key === 'ArrowRight'
        ? Math.min(links.length - 1, i + 1)
        : e.key === 'ArrowUp' || e.key === 'ArrowLeft'
          ? Math.max(0, i - 1)
          : e.key === 'Home'
            ? 0
            : e.key === 'End'
              ? links.length - 1
              : null
    if (next === null) return
    e.preventDefault()
    links[next].focus()
  }

  return (
    <nav className="rail" aria-label={label} onKeyDown={onKeyDown}>
      <ol className="rail-list">
        {items.map((item, i) => {
          const current = item.id === selected
          return (
            <li key={item.id}>
              <Link className="rail-item" to={to(item.id)} aria-current={current ? 'true' : undefined} replace>
                {item.thumb && (
                  <span className="rail-thumb">
                    <img src={dataUrl(item.thumb.src)} width={item.thumb.width} height={item.thumb.height} alt={item.thumb.alt} loading="lazy" />
                  </span>
                )}
                <span className="rail-text">
                  <span className="rail-index num" aria-hidden="true">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="rail-label">{item.label}</span>
                  <span className="rail-detail">{item.detail}</span>
                </span>
              </Link>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
