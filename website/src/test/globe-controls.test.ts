import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { viewFor } from '../components/marsglobe/projection'
import { useGlobeControls } from '../components/marsglobe/useGlobeControls'

const size = { width: 1200, height: 800 }

function pointer(target: Element, extra: Partial<{ clientX: number; clientY: number; timeStamp: number }> = {}) {
  const capture = vi.fn()
  return {
    capture,
    event: {
      button: 0,
      pointerId: 1,
      clientX: 0,
      clientY: 0,
      timeStamp: 0,
      target,
      currentTarget: { setPointerCapture: capture },
      ...extra,
    } as unknown as React.PointerEvent,
  }
}

describe('globe controls', () => {
  it('a press on a button (pin or callout) is not captured as a drag', () => {
    const { result } = renderHook(() => useGlobeControls({ initial: viewFor(0, 0, 1), size, reduced: true, autoRotate: false }))
    const button = document.createElement('button')
    const { event, capture } = pointer(button)
    act(() => result.current.handlers.onPointerDown(event))
    expect(capture).not.toHaveBeenCalled()
  })

  it('a press on the planet starts a drag that turns the globe under the pointer', () => {
    const { result } = renderHook(() => useGlobeControls({ initial: viewFor(0, 0, 1), size, reduced: true, autoRotate: false }))
    const stage = document.createElement('div')
    const down = pointer(stage)
    act(() => result.current.handlers.onPointerDown(down.event))
    expect(down.capture).toHaveBeenCalled()
    const before = result.current.view.yaw
    act(() => result.current.handlers.onPointerMove(pointer(stage, { clientX: 100, timeStamp: 16 }).event))
    // 100 px at a 288 px planet radius (0.36 × 800) turns the globe 100 / 288 rad.
    expect(result.current.view.yaw - before).toBeCloseTo(100 / 288, 6)
  })

  it('under reduced motion, a fly-to jumps straight to the target', () => {
    const { result } = renderHook(() => useGlobeControls({ initial: viewFor(0, 0, 1), size, reduced: true, autoRotate: false }))
    const target = viewFor(-47.0121, 18.7455, 60)
    act(() => result.current.flyTo(target))
    expect(result.current.view.pitch).toBeCloseTo(target.pitch, 9)
    expect(result.current.view.zoom).toBe(60)
  })

  it('keyboard: arrows turn, + and − zoom', () => {
    const { result } = renderHook(() => useGlobeControls({ initial: viewFor(0, 0, 2), size, reduced: true, autoRotate: false }))
    const key = (k: string) => ({ key: k, altKey: false, ctrlKey: false, metaKey: false, preventDefault: () => {} }) as unknown as React.KeyboardEvent
    const yaw = result.current.view.yaw
    act(() => void result.current.onKeyDown(key('ArrowLeft')))
    expect(result.current.view.yaw).toBeGreaterThan(yaw)
    act(() => void result.current.onKeyDown(key('+')))
    expect(result.current.view.zoom).toBeCloseTo(3)
    act(() => void result.current.onKeyDown(key('-')))
    expect(result.current.view.zoom).toBeCloseTo(2)
  })
})
