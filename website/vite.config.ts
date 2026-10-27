import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Relative base so the built site works from any static host or subpath.
export default defineConfig({
  base: './',
  plugins: [react()],
  // three.js is lazy-loaded; pre-bundle it with React so the dev server never serves two React copies.
  resolve: { dedupe: ['react', 'react-dom'] },
  optimizeDeps: { include: ['three', '@react-three/fiber'] },
  // The lazy GlobeCanvas chunk carries three.js (~245 kB gzipped); it loads only when motion is allowed.
  build: { chunkSizeWarningLimit: 1000 },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
