from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(title="AI Ops Observability Copilot")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
