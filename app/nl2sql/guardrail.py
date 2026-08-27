"""SQL guardrail: the app-level check on Claude-generated SQL before it ever
reaches the database.

This is NOT the only defense. The connection this SQL executes against is
also opened read_only=True with enable_external_access=false (app.db.
get_connection) — a physical backstop that survives a bug in this file,
verified during Epic 1-2's review: enable_external_access=False blocks
COPY TO / read_csv_auto / ATTACH / etc even on a read_only connection, and
read_only itself blocks INSERT/UPDATE/DELETE/DDL at the DuckDB level. Never
trust either layer alone — that lesson is why both exist. It's also why the
table allowlist below has been wrong twice already (round 1: a regex only
caught the first identifier after FROM/JOIN, missing comma joins; round 2:
CTE names were tracked as one flat, unscoped set, so declaring a throwaway
"WITH query_log AS (...)" whitelisted every real reference to that name
anywhere else in the query) — each time, `read_only` + external-access-off
meant the actual damage was "reads a table it shouldn't", not "escapes the
database entirely".

Uses DuckDB's own `json_serialize_sql()` to get the real parsed AST rather
than pattern-matching text — a regex cannot safely enumerate "every table
this query touches". Non-SELECT statements are rejected by the parser
itself (it refuses to serialize anything but a SELECT); multi-statement
strings show up as multiple parsed statements.

Known gap, not closed: this only examines data SOURCES (tables/CTEs/table
functions/SHOW-as-source). Scalar functions in the SELECT list or WHERE
clause aren't inspected at all, so `SELECT current_setting('memory_limit')`
passes and discloses this guardrail's own configuration — a minor
information leak (this project's own connection settings, not real data or
the filesystem), not currently actionable further since `getenv()` doesn't
exist in this DuckDB version and file-reading scalar functions are already
blocked by the binder for a reason unrelated to this file
(enable_external_access, see app.db). Worth an allowlist on scalar function
names if DuckDB ever adds a scalar with real read access (Epic 5 review
round 2, #8).
"""
import json

import duckdb

from app.db import QUERYABLE_TABLES

DEFAULT_ROW_LIMIT = 500

# Every table-reference node type DuckDB 1.1.3's parser can produce from a
# parseable SELECT (enumerated exhaustively during Epic 5 review round 3 by
# serializing ~40 constructs and structurally detecting every table-ref
# dict). Anything encountered that ISN'T one of these is rejected outright
# (see the "unrecognized table reference" check in _walk) — the walker used
# to silently pass through any node type it didn't explicitly recognize,
# which is exactly the posture that produced round 1's regex gap and round
# 2's SHOW_REF gap. Failing closed on an unknown type is the version of this
# check that survives a DuckDB upgrade adding a ninth type.
_SAFE_TABLEREF_TYPES = frozenset({"SUBQUERY", "JOIN", "EXPRESSION_LIST", "EMPTY", "PIVOT"})
_REJECTED_TABLEREF_TYPES = frozenset({"TABLE_FUNCTION", "SHOW_REF"})
# BASE_TABLE is handled separately (it needs the allowlist/CTE logic, not a
# blanket accept/reject).

# Table functions that only ever produce computed/literal rows — no file,
# network, or catalog access under any arguments — so they're safe to allow
# even though every other table function is rejected outright. Recovers
# ordinary time-bucketing SQL ("generate_series(min_ts, max_ts, INTERVAL 1
# HOUR)") that a model reaches for naturally (Epic 5 review round 3, nit #5).
_SAFE_TABLE_FUNCTIONS = frozenset({"range", "generate_series", "unnest"})


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
    # /ask requests (Epic 5 review round 1 follow-up). In-memory
    # connect/close has no file I/O and is cheap enough to pay per call.
    escaped = sql.replace("'", "''")
    conn = duckdb.connect(":memory:")
    try:
        raw = conn.execute(f"SELECT json_serialize_sql('{escaped}')").fetchone()[0]
    except duckdb.Error as e:
        raise GuardrailRejection(f"could not parse query: {e}") from e
    finally:
        conn.close()
    return json.loads(raw)


def _walk(node, visible_ctes: frozenset, disallowed_tables: set, disallowed_sources: set, all_real_tables: set) -> None:
    """Recursively walk the AST, checking every table-like reference AS IT'S
    FOUND rather than collecting names into flat sets and reconciling
    afterward — the flat-set approach is what caused round 2's CTE-shadowing
    bug (a CTE name defined in one subquery incorrectly whitelisted a real
    table reference in an unrelated sibling subquery, since "is this name a
    CTE" was answered against one global set instead of the CTE names
    actually in scope at that point in the tree).

    visible_ctes is the set of CTE names in scope at THIS node — extended
    only for this node's own subtree when a cte_map is present, never
    globally. A BASE_TABLE only gets the CTE exemption if it is unqualified
    (no schema_name/catalog_name) AND its name is in visible_ctes — a
    qualified reference (main.query_log, memory.main.events) can never
    resolve to a CTE regardless of name, so it's always checked directly.

    SHOW_REF (the node type for `SHOW t` / `DESCRIBE t` / `SUMMARIZE t` used
    as a data source) is rejected unconditionally, same as TABLE_FUNCTION —
    SUMMARIZE in particular returns real per-column min/max/quartiles, not
    just metadata, so it's a data leak, not a schema-disclosure nit.
    """
    if isinstance(node, dict):
        node_type = node.get("type")

        if node_type == "BASE_TABLE":
            name = node.get("table_name", "").lower()
            qualified = bool(node.get("schema_name")) or bool(node.get("catalog_name"))
            if qualified:
                # The app never needs a qualified reference — schema_prompt
                # only ever names bare table names — so reject outright
                # rather than trying to validate "is this qualifier really
                # this database's own main schema", which is one more name
                # to get wrong (Epic 5 review round 3, #3: `memory.main.
                # events` passed because only the bare last component was
                # checked).
                all_real_tables.add(name)
                disallowed_tables.add(
                    f"{node.get('catalog_name', '')}.{node.get('schema_name', '')}.{name}".strip(".")
                )
            else:
                is_cte_reference = name in visible_ctes
                if not is_cte_reference:
                    all_real_tables.add(name)
                    if name not in QUERYABLE_TABLES:
                        disallowed_tables.add(name)
        elif node_type == "TABLE_FUNCTION":
            fn = node.get("function", {})
            fn_name = fn.get("function_name", "<unknown function>")
            if fn_name.lower() not in _SAFE_TABLE_FUNCTIONS:
                disallowed_sources.add(fn_name)
        elif node_type == "SHOW_REF":
            disallowed_sources.add(f"SHOW/DESCRIBE/SUMMARIZE: {node.get('table_name', '?')}")
        elif (
            isinstance(node_type, str)
            and node_type not in _SAFE_TABLEREF_TYPES
            and node_type not in _REJECTED_TABLEREF_TYPES
            and "sample" in node
            and "alias" in node
            and "class" not in node
        ):
            # A dict shaped like a table reference (every real one carries
            # both `sample` and `alias`, and never `class` — expression
            # nodes always carry `class`; `type` on a non-table-ref node can
            # also be a nested dict rather than a string, e.g. a literal's
            # value type, so the isinstance check comes first) but of a type
            # this walker has never seen: fail closed rather than silently
            # allowing it through (Epic 5 review round 3, #1 — validated
            # against the full enumerated corpus: 8/8 known types matched,
            # zero false positives on ordinary expression nodes).
            disallowed_sources.add(f"unrecognized table reference: {node_type}")

        local_ctes = visible_ctes
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict) and cte_map.get("map"):
            new_names = {e["key"].lower() for e in cte_map["map"] if e.get("key")}
            local_ctes = visible_ctes | new_names
            # Walk each CTE's own body with the extended scope too — DuckDB
            # allows a later CTE in the same WITH clause to reference an
            # earlier one (and, harmlessly for this check, a self-reference
            # for a recursive CTE).
            for entry in cte_map["map"]:
                body = entry.get("value", {}).get("query")
                if body is not None:
                    _walk(body, local_ctes, disallowed_tables, disallowed_sources, all_real_tables)

        for key, value in node.items():
            if key == "cte_map":
                continue  # already walked above with the correctly extended scope
            _walk(value, local_ctes, disallowed_tables, disallowed_sources, all_real_tables)
    elif isinstance(node, list):
        for item in node:
            _walk(item, visible_ctes, disallowed_tables, disallowed_sources, all_real_tables)


def validate_and_prepare(sql: str) -> str:
    """Returns SQL that is safe to execute. The returned SQL is ALWAYS
    wrapped in an outer `SELECT * FROM (<original>) LIMIT n` rather than
    having a LIMIT string-appended — round 1 found the appended form is
    defeated by a trailing `--` comment, a `LIMIT` inside a string literal
    satisfying the presence check, or an inner subquery's own LIMIT. A
    structural outer wrapper is immune to all three since it's a completely
    separate clause the model's text never touches. The newline before the
    closing paren is load-bearing, not style — see the comment at the return
    statement."""
    if not sql or not sql.strip():
        raise GuardrailRejection("empty query")

    ast = _serialize_to_ast(sql)
    if ast.get("error"):
        raise GuardrailRejection(f"not a valid single SELECT statement: {ast.get('error_message', '')}")

    statements = ast.get("statements", [])
    if len(statements) != 1:
        raise GuardrailRejection(f"expected exactly one statement, got {len(statements)}")

    disallowed_tables: set[str] = set()
    disallowed_sources: set[str] = set()
    all_real_tables: set[str] = set()
    _walk(statements[0], frozenset(), disallowed_tables, disallowed_sources, all_real_tables)

    if disallowed_sources:
        raise GuardrailRejection(f"query uses a disallowed data source: {sorted(disallowed_sources)}")
    if disallowed_tables:
        raise GuardrailRejection(f"query references a table that is not allowed: {sorted(disallowed_tables)}")
    if not all_real_tables:
        # e.g. a bare "SELECT 1" — harmless but useless here, reject rather
        # than silently no-op.
        raise GuardrailRejection("query does not reference an allowed table")

    inner = sql.strip().rstrip(";")
    # The newline before the closing paren is load-bearing: if the model's
    # query ends with a `--` line comment, appending `) AS ... LIMIT` on the
    # SAME line would itself land inside that comment — the exact class of
    # bug this wrapper exists to close, one level up. A newline ends the
    # comment first (verified: without it, a trailing `--` comment produces
    # a syntax error here instead of silently losing the LIMIT).
    return f"SELECT * FROM (\n{inner}\n) AS _guardrail_wrapped LIMIT {DEFAULT_ROW_LIMIT}"
