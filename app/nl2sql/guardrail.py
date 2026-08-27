"""SQL guardrail: the app-level check on Claude-generated SQL before it ever
reaches the database.

This is NOT the only defense. The connection this SQL executes against is
also opened read_only=True with enable_external_access=false (app.db.
get_connection) — a physical backstop that survives a bug in this file,
verified during Epic 1-2's review: enable_external_access=False blocks
COPY TO / read_csv_auto / ATTACH / etc even on a read_only connection, and
read_only itself blocks INSERT/UPDATE/DELETE/DDL at the DuckDB level. Never
trust either layer alone — that lesson is why both exist.
"""
import re

from app.db import QUERYABLE_TABLES

_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|COPY|PRAGMA|EXPORT|IMPORT|"
    r"CALL|SET|GRANT|REVOKE|VACUUM|CHECKPOINT|INSTALL|LOAD)\b",
    re.IGNORECASE,
)
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_STARTS_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

DEFAULT_ROW_LIMIT = 500


class GuardrailRejection(Exception):
    """Raised when generated SQL fails the guardrail. The message is safe to
    return to a client — it never echoes the rejected SQL, so a rejection
    response can't be used to probe the guardrail's exact boundaries."""


def validate_and_prepare(sql: str) -> str:
    """Returns SQL that is safe to execute (with a LIMIT injected if the
    model didn't include one), or raises GuardrailRejection."""
    if not sql or not sql.strip():
        raise GuardrailRejection("empty query")

    stripped = sql.strip()

    # A single statement, no semicolon-chaining. Do this before anything
    # else — a trailing semicolon is the one case we tolerate (strip it),
    # anything with an EMBEDDED semicolon is a multi-statement attempt.
    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if ";" in stripped:
        raise GuardrailRejection("only a single statement is allowed")

    if not _STARTS_SELECT.match(stripped):
        raise GuardrailRejection("only SELECT statements are allowed")

    if _DISALLOWED_KEYWORDS.search(stripped):
        raise GuardrailRejection("query contains a disallowed keyword")

    referenced = {m.group(1).lower() for m in _TABLE_REF.finditer(stripped)}
    disallowed = referenced - QUERYABLE_TABLES
    if disallowed:
        raise GuardrailRejection("query references a table that is not allowed")
    if not referenced:
        # A SELECT with no FROM at all (e.g. "SELECT 1") is harmless but
        # also useless for this app — reject rather than silently no-op.
        raise GuardrailRejection("query does not reference an allowed table")

    if not _HAS_LIMIT.search(stripped):
        stripped = f"{stripped} LIMIT {DEFAULT_ROW_LIMIT}"

    return stripped
