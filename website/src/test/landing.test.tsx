import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

function renderLanding() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <App />
    </MemoryRouter>,
  )
}

describe('landing', () => {
  it('offers both worlds in fixed order, each themed by its own tokens', () => {
    renderLanding()
    const doors = within(screen.getByRole('list', { name: 'Worlds' })).getAllByRole('link')
    expect(doors.map((d) => d.getAttribute('href'))).toEqual(['/mars', '/moon'])
    expect(doors.map((d) => d.getAttribute('data-world'))).toEqual(['mars', 'moon'])
    expect(document.querySelector('[data-winner]')).toBeNull()
  })

  it('shows counts taken from the exported data', () => {
    renderLanding()
    const [mars, moon] = within(screen.getByRole('list', { name: 'Worlds' })).getAllByRole('link')
    expect(mars).toHaveTextContent('15,233 tiles')
    expect(mars).toHaveTextContent('5 sites')
    expect(moon).toHaveTextContent('779 test crops')
    expect(moon).toHaveTextContent('3 classes')
  })

  it('links to the caveats and credits the imagery', () => {
    renderLanding()
    expect(screen.getByRole('link', { name: 'Read the caveats' })).toHaveAttribute('href', '/about')
    expect(screen.getByText(/NASA\/JPL\/University of Arizona/)).toBeInTheDocument()
  })

  it('gives every door image real alt text', () => {
    renderLanding()
    for (const img of screen.getAllByRole('img')) expect(img.getAttribute('alt')?.length).toBeGreaterThan(10)
  })
})
