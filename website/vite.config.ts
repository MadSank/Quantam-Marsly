import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Relative base so the built site works from any static host or subpath.
export default defineConfig({
  base: './',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
