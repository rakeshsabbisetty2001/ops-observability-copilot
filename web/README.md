# AI Ops Observability Copilot — frontend

React/TypeScript SPA replacing the project's original Streamlit UI. Same three views (Ask, Browse, Drill-down) against the same FastAPI backend (`../app/`), no new endpoints — see `src/api.ts` for the exact contract.

## Run locally

```bash
npm install
npm run dev          # http://localhost:5173, talks to VITE_API_URL (defaults to http://localhost:8000)
```

Backend must have `FRONTEND_ORIGIN` unset or including `http://localhost:5173` (`app/main.py`'s CORS config always allows the Vite dev origin regardless).

## Build

```bash
npm run build         # tsc -b && vite build → dist/
```

## Deploy

Vercel, root directory `web/`, env var `VITE_API_URL` set to the deployed Render API URL. Once live, the API's `FRONTEND_ORIGIN` env var (Render dashboard) needs to be set to the Vercel URL for CORS to allow it.
