import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App.tsx'

describe('App', () => {
  it('renders the project title', () => {
    window.history.pushState({}, '', '/nowthennow/')
    render(<App />)
    expect(screen.getByRole('heading', { name: 'now then now' })).toBeInTheDocument()
  })

  it('links to the github repo on /about', () => {
    window.history.pushState({}, '', '/nowthennow/about')
    render(<App />)
    expect(screen.getByRole('link', { name: 'github.com/ch345/nowthennow' })).toHaveAttribute(
      'href',
      'https://github.com/ch345/nowthennow',
    )
  })
})
