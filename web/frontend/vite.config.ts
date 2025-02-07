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
        path.resolve(__dirname, 'public'),
      ],
    },
    host: true, // Allow connections from all IPs
    watch: {
      usePolling: true, // Enable polling for remote development
    },
  },
  resolve: {
    alias: {
      '@fonts': path.resolve(__dirname, 'public/fonts'),
    },
  },
  publicDir: 'public',
  optimizeDeps: {
    include: ['@fontsource/roboto'],
  },
})
