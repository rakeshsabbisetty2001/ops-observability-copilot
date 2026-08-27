"""Text-to-SQL generation: one Claude call turning a natural-language
question into a single SQL statement. Returns raw, UNVALIDATED SQL — every
caller must run it through app.nl2sql.guardrail before executing it."""
import anthropic

from app.config import settings
from app.nl2sql.schema_prompt import SCHEMA_DESCRIPTION

_SYSTEM_PROMPT = (
    "You translate a natural-language question into exactly one SQL SELECT "
    "statement for DuckDB. Output ONLY the SQL text — no prose, no markdown "
    "code fences, no explanation.\n\n" + SCHEMA_DESCRIPTION
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_sql(question: str) -> str:
    response = _get_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = response.content[0].text.strip()
    # Despite the instruction, models sometimes wrap output in a fence anyway.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("sql"):
            text = text[3:]
        text = text.strip()
    return text
