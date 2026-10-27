import { createContext, useContext } from 'react'

export type World = 'mars' | 'moon'

export const WORLDS: readonly World[] = ['mars', 'moon']

export const WorldContext = createContext<World | null>(null)

/** The current world, or null on the neutral shell pages (landing, About). */
export function useWorld(): World | null {
  return useContext(WorldContext)
}
