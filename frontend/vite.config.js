import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config — serves the app on localhost:5173 by default.
// The `proxy` block forwards /api/* requests to the Flask backend
// (localhost:5000) so you never deal with CORS during development.
// When you build the Flask backend, just change the target port here.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // All requests starting with /api are forwarded to the Flask backend.
      // Change the target URL when your backend is running on a different port.
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
