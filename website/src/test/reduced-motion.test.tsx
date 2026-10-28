import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AnimatedNumber } from '../components/AnimatedNumber'
import { MotionContext, type MotionState } from '../components/motionContext'

const fmt = (v: number | null) => (v === null ? '—' : v.toFixed(1))
const painted = (el: HTMLElement) => el.querySelector('[aria-hidden="true"]')!.textContent

function withMotion(reduced: boolean, value: number) {
  const state: MotionState = { allowed: !reduced, reduced, paused: false, setPaused: () => {} }
  return (
    <MotionContext.Provider value={state}>
      <AnimatedNumber value={value} format={fmt} />
    </MotionContext.Provider>
  )
}

describe('reduced motion', () => {
  it('AnimatedNumber renders the final value immediately', () => {
    const { container, rerender } = render(withMotion(true, 1))
    expect(painted(container)).toBe('1.0')
    rerender(withMotion(true, 42))
    expect(painted(container)).toBe('42.0')
  })

  it('with motion allowed it starts from the previous value, never flashing the final one', () => {
    const { container, rerender } = render(withMotion(false, 1))
    expect(painted(container)).toBe('1.0')
    rerender(withMotion(false, 42))
    expect(painted(container)).toBe('1.0')
  })

  it('always exposes the final value to assistive technology', () => {
    const { container, rerender } = render(withMotion(false, 1))
    rerender(withMotion(false, 42))
    expect(container.querySelector('.visually-hidden')!.textContent).toBe('42.0')
  })
})
