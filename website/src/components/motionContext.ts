import { createContext, useContext } from 'react'

export interface MotionState {
  /** True when ambient motion may run: no reduced-motion preference and not paused. */
  allowed: boolean
  reduced: boolean
  paused: boolean
  setPaused: (paused: boolean) => void
}

export const MotionContext = createContext<MotionState>({ allowed: false, reduced: true, paused: false, setPaused: () => {} })

export function useMotion(): MotionState {
  return useContext(MotionContext)
}
