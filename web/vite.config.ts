import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // app/main.py's CORS allowlist hardcodes http://localhost:5173 for dev.
    // Without this, a busy port (e.g. another Vite project already running)
    // makes Vite silently move to 5174+, which the backend then blocks —
    // surfacing as "could not reach the API", indistinguishable from a
    // genuinely down API (review round 1, N5). Failing loudly beats a
    // silent port change here.
    strictPort: true,
  },
})
