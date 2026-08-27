"""SQL guardrail: the app-level check on Claude-generated SQL before it ever
reaches the database.

This is NOT the only defense. The connection this SQL executes against is
also opened read_only=True with enable_external_access=false (app.db.
get_connection) — a physical backstop that survives a bug in this file,
verified during Epic 1-2's review: enable_external_access=False blocks
COPY TO / read_csv_auto / ATTACH / etc even on a read_only connection, and
read_only itself blocks INSERT/UPDATE/DELETE/DDL at the DuckDB level. Never
trust either layer alone — that lesson is why both exist.

Round 1 of this epic's review shipped a regex-based version of this file and
found it had a working bypass: `_TABLE_REF` only captured the first
identifier after FROM/JOIN, so a comma join (`FROM events, query_log`) or a
comment-obfuscated JOIN (`JOIN/**/query_log`) leaked the audit log and
DuckDB's own catalog functions. Regexes cannot safely enumerate "every table
this query touches" — only a real parser can. This version uses DuckDB's own
`json_serialize_sql()` to get the actual parsed AST and walks it for real
table references, which also means: non-SELECT statements are rejected by
the parser itself (it refuses to serialize anything but a SELECT), multi-
statement strings are visible as multiple parsed statements, comments and
string-literal contents can no longer be mistaken for SQL syntax, and CTEs
(WITH ...) are supported since their table references are enumerable too.
"""
import json

import duckdb

from app.db import QUERYABLE_TABLES

DEFAULT_ROW_LIMIT = 500


class GuardrailRejection(Exception):
    """Raised when generated SQL fails the guardrail. The message is a
    single fixed string at every call site that returns it to a client
    (see app.nl2sql.ask) — never the rejected SQL or the specific reason,
    which would otherwise make a rejection response an oracle for mapping
    the guardrail's exact boundaries. The specific reason is still
    available via .args for logging to query_log, where it belongs."""


def _serialize_to_ast(sql: str) -> dict:
    # A fresh in-memory connection per call, not a shared module-level one:
    # DuckDB connections are not safe for concurrent execute() calls from
    # multiple threads — a shared parser connection caused a real access
    # violation (native crash, not a Python exception) under 8 concurrent
    # /ask requests (Epic 5 review round 1 follow-up, found while verifying
    # the fix for finding #3's concurrency issues). In-memory connect/close
    # has no file I/O and is cheap enough to pay per call.
    escaped = sql.replace("'", "''")
    conn = duckdb.connect(":memory:")
    try:
        raw = conn.execute(f"SELECT json_serialize_sql('{escaped}')").fetchone()[0]
    except duckdb.Error as e:
        raise GuardrailRejection(f"could not parse query: {e}") from e
    finally:
        conn.close()
    return json.loads(raw)


def _collect_tables_and_ctes(node, tables: set[str], cte_names: set[str], table_functions: set[str]) -> None:
    """Recursively walk the AST dict/list structure.
    - type == BASE_TABLE names a real table reference.
    - type == TABLE_FUNCTION is a call like duckdb_settings() or
      read_csv_auto('x') used as a data source — collected separately and
      ALWAYS rejected (no allowlist for these at all): the app has no
      legitimate use for any function-based table source, and this project
      already found duckdb_settings()/duckdb_databases()/duckdb_tables()
      dumping the server's config and on-disk paths when only BASE_TABLE
      nodes were checked (Epic 5 review round 1, #1).
    - cte_map collects locally-defined CTE names, which are not real tables
      and must not be checked against the allowlist (a query's own
      "WITH x AS (...)" defines a name that only exists inside that query).
    """
    if isinstance(node, dict):
        if node.get("type") == "BASE_TABLE":
            tables.add(node.get("table_name", "").lower())
        elif node.get("type") == "TABLE_FUNCTION":
            fn = node.get("function", {})
            table_functions.add(fn.get("function_name", "<unknown>"))
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict):
            for entry in cte_map.get("map", []):
                key = entry.get("key")
                if key:
                    cte_names.add(key.lower())
        for value in node.values():
            _collect_tables_and_ctes(value, tables, cte_names, table_functions)
    elif isinstance(node, list):
        for item in node:
            _collect_tables_and_ctes(item, tables, cte_names, table_functions)


def validate_and_prepare(sql: str) -> str:
    """Returns SQL that is safe to execute. The returned SQL is ALWAYS
    wrapped in an outer `SELECT * FROM (<original>) LIMIT n` rather than
    having a LIMIT string-appended — round 1 found the appended form is
    defeated by a trailing `--` comment, a `LIMIT` inside a string literal
    satisfying the presence check, or an inner subquery's own LIMIT. A
    structural outer wrapper is immune to all three since it's a completely
    separate clause the model's text never touches."""
    if not sql or not sql.strip():
        raise GuardrailRejection("empty query")

    ast = _serialize_to_ast(sql)
    if ast.get("error"):
        raise GuardrailRejection(f"not a valid single SELECT statement: {ast.get('error_message', '')}")

    statements = ast.get("statements", [])
    if len(statements) != 1:
        raise GuardrailRejection(f"expected exactly one statement, got {len(statements)}")

    tables: set[str] = set()
    cte_names: set[str] = set()
    table_functions: set[str] = set()
    _collect_tables_and_ctes(statements[0], tables, cte_names, table_functions)

    if table_functions:
        raise GuardrailRejection(f"query uses a function as a data source, which is never allowed: {sorted(table_functions)}")

    real_tables = tables - cte_names
    disallowed = real_tables - QUERYABLE_TABLES
    if disallowed:
        raise GuardrailRejection(f"query references a table that is not allowed: {sorted(disallowed)}")
    if not real_tables:
        raise GuardrailRejection("query does not reference an allowed table")

    inner = sql.strip().rstrip(";")
    # The newline before the closing paren is load-bearing, not style: if the
    # model's query ends with a `--` line comment, appending `) AS ... LIMIT`
    # on the SAME line would itself land inside that comment — the exact
    # class of bug this wrapper exists to close, one level up. A newline
    # ends the comment first (verified: without it, a trailing `--` comment
    # produces a syntax error here instead of silently losing the LIMIT,
    # which is a symptom of the same root cause, not a different bug).
    return f"SELECT * FROM (\n{inner}\n) AS _guardrail_wrapped LIMIT {DEFAULT_ROW_LIMIT}"
