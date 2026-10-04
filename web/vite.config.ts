import { cpSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vitest/config'

function githubPagesAbout(): Plugin {
  return {
    name: 'github-pages-about',
    apply: 'build',
    closeBundle() {
      const aboutDir = resolve('dist/about')
      mkdirSync(aboutDir, { recursive: true })
      cpSync(resolve('dist/index.html'), resolve(aboutDir, 'index.html'))
    },
  }
}

export default defineConfig({
  // GitHub Pages serves the site from /<repo>/
  base: '/nowthennow/',
  plugins: [react(), githubPagesAbout()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
