import { useEffect, useState } from 'react'
import { dataUrl } from './dataUrl'

const cache = new Map<string, Promise<unknown>>()

export interface JsonState<T> {
  data: T | null
  error: boolean
}

/** Fetch a JSON file under public/data once per session. A failed load is retried on the next mount. */
export function useJson<T>(rel: string | null): JsonState<T> {
  const [state, setState] = useState<{ rel: string; data: T | null; error: boolean } | null>(null)

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
      (data) => live && setState({ rel, data: data as T, error: false }),
      () => live && setState({ rel, data: null, error: true }),
    )
    return () => {
      live = false
    }
  }, [rel])

  return state && state.rel === rel ? { data: state.data, error: state.error } : { data: null, error: false }
}
