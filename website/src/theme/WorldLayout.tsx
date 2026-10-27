import { useEffect, type ReactNode } from 'react'
import { WorldContext, type World } from './world'

/**
 * Applies a world's design tokens to its subtree (via data-world) and to <html>,
 * so the page background matches while the world is shown.
 */
export function WorldLayout({ world, children }: { world: World; children: ReactNode }) {
  useEffect(() => {
    const root = document.documentElement
    root.dataset.world = world
    return () => {
      delete root.dataset.world
    }
  }, [world])

  return (
    <WorldContext.Provider value={world}>
      <div className="world-root" data-world={world}>
        {children}
      </div>
    </WorldContext.Provider>
  )
}
