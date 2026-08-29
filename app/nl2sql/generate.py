"""Text-to-SQL generation: one Claude call turning a natural-language
question into a single SQL statement. Returns raw, UNVALIDATED SQL — every
caller must run it through app.nl2sql.guardrail before executing it."""
import re

import anthropic

from app.config import settings
from app.nl2sql.schema_prompt import SCHEMA_DESCRIPTION

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

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
    # content[0] is NOT reliably the text block — claude-sonnet-5 puts a
    # `thinking` block (text=None) first when thinking is enabled by
    # default, which crashed every real call with AttributeError before
    # this fix. Never caught locally: every test mocks this call, and this
    # was the first real end-to-end /ask request against a live key
    # (first Render deploy). Find the actual text block instead of
    # assuming position.
    text = next(block.text for block in response.content if block.type == "text").strip()
    # Despite the instruction, models sometimes wrap output in a fence anyway
    # — any language tag (```sql, ```SQL, ```duckdb, bare ```), not just
    # "sql" specifically (Epic 5 review round 1, #7: a plausible ```duckdb
    # tag survived the old strip("`") + literal "sql"-prefix check intact).
    text = _FENCE.sub("", text).strip()
    return text
