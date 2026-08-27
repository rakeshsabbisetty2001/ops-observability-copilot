"""Plain-English summary of a SQL result, grounded in the actual returned
rows — the system prompt explicitly forbids inventing numbers not present in
the data, since this answer is what a non-technical user reads."""
import anthropic

from app.config import settings

_SYSTEM_PROMPT = (
    "Answer the user's question in one or two plain-English sentences, using "
    "only the data given below. Never state a number, date, or fact that "
    "isn't directly present in that data."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def summarize_result(question: str, rows: list[dict]) -> str:
    if not rows:
        return "No matching data was found for that question."

    response = _get_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\nData: {rows}"}],
    )
    return response.content[0].text.strip()
