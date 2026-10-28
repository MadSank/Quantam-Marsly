import { useEffect, useState } from 'react'
import { dataUrl } from './dataUrl'

const cache = new Map<string, Promise<unknown>>()

/** Fetch a JSON file under public/data once per session; null while loading or on failure. */
export function useJson<T>(rel: string | null): T | null {
  const [state, setState] = useState<{ rel: string; data: T } | null>(null)

  useEffect(() => {
    if (!rel) return
    let live = true
    let request = cache.get(rel)
    if (!request) {
      request = fetch(dataUrl(rel)).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${rel}`))))
      request.catch(() => cache.delete(rel))
      cache.set(rel, request)
    }
    request.then(
      (data) => live && setState({ rel, data: data as T }),
      () => {},
    )
    return () => {
      live = false
    }
  }, [rel])

  return state && state.rel === rel ? state.data : null
}
