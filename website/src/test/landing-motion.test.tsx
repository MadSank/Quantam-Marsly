import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { rotationForLongitude, surfacePoint } from '../components/globe/spec'

function mockReducedMotion(reduce: boolean) {
  vi.stubGlobal(
    'matchMedia',
    (query: string) =>
      ({
        matches: reduce && query.includes('reduce'),
        media: query,
        addEventListener: () => {},
        removeEventListener: () => {},
      }) as unknown as MediaQueryList,
  )
}

function renderLanding() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <App />
    </MemoryRouter>,
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
  try {
    localStorage.clear()
  } catch {
    // ignore
  }
})

describe('landing motion', () => {
  it('shows a pause control that stops ambient motion', () => {
    mockReducedMotion(false)
    const { container } = renderLanding()
    const toggle = screen.getByRole('button', { name: 'Pause motion' })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    expect(container.querySelector('.landing')).toHaveAttribute('data-motion', 'on')
    fireEvent.click(toggle)
    expect(screen.getByRole('button', { name: 'Play motion' })).toHaveAttribute('aria-pressed', 'true')
    expect(container.querySelector('.landing')).toHaveAttribute('data-motion', 'off')
  })

  it('under reduced motion: no control, motion off, stills only', () => {
    mockReducedMotion(true)
    const { container } = renderLanding()
    expect(screen.queryByRole('button', { name: /motion/ })).toBeNull()
    expect(container.querySelector('.landing')).toHaveAttribute('data-motion', 'off')
    expect(container.querySelectorAll('.globe-still')).toHaveLength(2)
    expect(container.querySelector('.globe-live')).toBeNull()
  })

  it('keeps the planets decorative and credits the globe mosaics', () => {
    mockReducedMotion(true)
    const { container } = renderLanding()
    expect(container.querySelector('.landing-sky')).toHaveAttribute('aria-hidden', 'true')
    expect(screen.getByText(/Viking MDIM 2\.1/)).toBeInTheDocument()
    expect(screen.getByText(/LROC WAC colour/)).toBeInTheDocument()
  })
})

describe('globe geometry', () => {
  it('brings the requested longitude to the front', () => {
    // A surface point at the front longitude, rotated about y, must face the camera (+z).
    for (const lon of [-70, 0, 120]) {
      const [x, , z] = surfacePoint(0, lon)
      const t = rotationForLongitude(lon)
      const xr = x * Math.cos(t) + z * Math.sin(t)
      const zr = -x * Math.sin(t) + z * Math.cos(t)
      expect(xr).toBeCloseTo(0, 9)
      expect(zr).toBeCloseTo(1, 9)
    }
  })
})
