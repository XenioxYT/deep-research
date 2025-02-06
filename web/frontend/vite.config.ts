import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      strict: false,
      allow: [
        '..',
        '../../node_modules/@fontsource',
      ],
    },
    host: true, // Allow connections from all IPs
    watch: {
      usePolling: true, // Enable polling for remote development
    },
  },
  optimizeDeps: {
    include: ['@fontsource/roboto'],
  },
})
