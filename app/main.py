from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.nl2sql.ask import ask

app = FastAPI(title="AI Ops Observability Copilot")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    question: str
    sql: str | None
    answer: str | None
    row_count: int | None
    anomaly_ids: list[int]
    error: str | None = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return AskResponse(**ask(request.question))
