import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ command }) => {
  // A module-scope `throw` in api.ts (tried first, review round 2's NEW-1)
  // compiles fine and ships — `npm run build` still exits 0, so a missing
  // VITE_API_URL becomes a completely blank deployed page instead of a
  // build failure. This is the only place that can actually fail the
  // build: process.env here is the shell's real env at build time, not the
  // inlined import.meta.env the browser bundle gets.
  if (command === "build" && !process.env.VITE_API_URL) {
    throw new Error("VITE_API_URL must be set for a production build (Vercel → Settings → Environment Variables)");
  }

  return {
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
  };
})
