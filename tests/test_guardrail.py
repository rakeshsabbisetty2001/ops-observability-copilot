import duckdb
import pytest

from app.nl2sql.guardrail import DEFAULT_ROW_LIMIT, GuardrailRejection, validate_and_prepare


def _exec(sql: str):
    """Actually run the prepared SQL against a throwaway table shaped like
    events — a wrapper that's merely well-formed as a string can still be
    syntactically broken once assembled (found exactly this way: a trailing
    `--` comment ate the wrapper's own closing paren)."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE events AS SELECT i AS id, 'x' AS message FROM range(1000) t(i)")
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


def test_plain_select_passes_and_gets_wrapped_with_a_limit():
    result = validate_and_prepare("SELECT * FROM events")
    assert result == f"SELECT * FROM (\nSELECT * FROM events\n) AS _guardrail_wrapped LIMIT {DEFAULT_ROW_LIMIT}"


def test_existing_limit_does_not_defeat_the_outer_wrap():
    """The outer LIMIT is structural, not string-appended — it applies
    regardless of what the inner query already does with LIMIT."""
    result = validate_and_prepare("SELECT * FROM events LIMIT 10")
    assert result.count("LIMIT") == 2
    assert result.endswith(f"LIMIT {DEFAULT_ROW_LIMIT}")


def test_trailing_semicolon_is_tolerated():
    result = validate_and_prepare("SELECT * FROM events;")
    assert result.count(";") == 0


def test_join_across_two_allowed_tables_passes():
    result = validate_and_prepare(
        "SELECT e.* FROM events e JOIN detected_anomalies d ON e.service = d.service"
    )
    assert "LIMIT" in result


def test_cte_over_allowed_tables_is_now_supported():
    """A regex-based guardrail could not safely tell a CTE name from a real
    table reference, so CTEs were rejected outright — the parser-based
    guardrail can enumerate a CTE's real table references, so legitimate
    multi-step queries (very common for exactly the aggregations this app
    is about) are no longer collateral damage."""
    result = validate_and_prepare("WITH recent AS (SELECT * FROM events) SELECT * FROM recent")
    assert "LIMIT" in result


# --- Epic 5 review round 1, #1: the working table-allowlist bypass ---------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM events, query_log",
        "SELECT q.question FROM events e, query_log q",
        "SELECT * FROM events JOIN/**/query_log ON 1=1",
        "SELECT * FROM events LEFT JOIN/**/query_log ON true",
        "SELECT * FROM events, main.query_log",
        "SELECT * FROM events, sqlite_master",
        "SELECT * FROM events, duckdb_settings()",
        "SELECT * FROM events, duckdb_databases()",
        "SELECT * FROM events, duckdb_tables()",
        "SELECT * FROM events, ground_truth_anomalies",
        "SELECT * FROM events, information_schema.tables",
    ],
)
def test_comma_join_and_comment_obfuscated_join_bypass_rejected(sql):
    """These all PASSED the round-1 regex guardrail and then executed,
    leaking query_log (every prior user's question + generated SQL) and
    DuckDB's own catalog. The parser-based guardrail walks the real AST for
    every base-table reference, not just the first identifier after a bare
    FROM/JOIN keyword, so a second table joined by comma or hidden behind a
    comment is no longer invisible."""
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


# --- Epic 5 review round 1, #2: the LIMIT-defeat cases ---------------------


def test_trailing_comment_cannot_defeat_the_limit():
    """A regex-based LIMIT check that string-appends 'LIMIT 500' after the
    model's text is defeated when the model's last line is a `--` comment —
    the appended limit lands inside the comment. The structural outer wrap
    can't be swallowed by anything inside the inner query's text — but only
    once the wrapper itself puts a newline before its own closing paren.
    Without that newline this exact input produced a syntax error (the
    comment ate the wrapper's `)` too), so this is checked by actually
    EXECUTING the result, not just inspecting its string shape."""
    result = validate_and_prepare("SELECT * FROM events -- give me everything")
    assert result.rstrip().endswith(f"LIMIT {DEFAULT_ROW_LIMIT}")
    df = _exec(result)
    assert len(df) == DEFAULT_ROW_LIMIT  # 1000 events in the fixture, capped correctly


def test_limit_inside_a_string_literal_cannot_satisfy_the_check():
    result = validate_and_prepare("SELECT * FROM events WHERE message = 'LIMIT 1'")
    assert result.endswith(f"LIMIT {DEFAULT_ROW_LIMIT}")
    df = _exec(result)
    assert len(df) == 0  # no event's message literally equals "LIMIT 1" in the fixture, but it must still run


def test_inner_subquery_limit_cannot_substitute_for_the_outer_one():
    result = validate_and_prepare("SELECT * FROM (SELECT * FROM events LIMIT 1) t")
    assert result.endswith(f"LIMIT {DEFAULT_ROW_LIMIT}")
    df = _exec(result)
    assert len(df) == 1  # the inner LIMIT 1 is real; the outer wrap doesn't need to override a smaller inner limit


# --- non-SELECT statements: rejected by DuckDB's own parser ----------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE events",
        "DELETE FROM events",
        "UPDATE events SET value = 0",
        "INSERT INTO events VALUES (1, now(), 'a', 'b', 1.0, 'info', '')",
        "CREATE TABLE evil AS SELECT * FROM events",
        "ALTER TABLE events ADD COLUMN x INT",
        "ATTACH 'other.duckdb' AS other",
        "PRAGMA database_list",
        "COPY events TO 'out.csv'",
        "SET enable_external_access=true",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL pragma_table_info('events')",
        "VACUUM",
        "CHECKPOINT",
        "EXPLAIN SELECT * FROM events",
    ],
)
def test_disallowed_statement_type_rejected(sql):
    """json_serialize_sql refuses to serialize anything but a SELECT, so
    every one of these is caught by DuckDB's own parser before this
    guardrail's table-allowlist logic ever runs — no keyword blocklist
    needed, which also means it can't false-positive on ordinary log text
    (see the removed keyword-blocklist tests this replaces)."""
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM query_log",
        "SELECT * FROM ground_truth_anomalies",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM pg_catalog.pg_tables",
    ],
)
def test_single_table_disallowed_reference_rejected(sql):
    """The ground-truth answer key must be unreachable through this path
    regardless of what the model is coerced into asking for — this is the
    guardrail half of the defense-in-depth story (the physical half is
    ground_truth_anomalies living in a separate DuckDB file entirely)."""
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "not sql at all",
        "SELECT 1",  # no FROM, no table reference — harmless but useless, reject rather than no-op
    ],
)
def test_malformed_or_tableless_query_rejected(sql):
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


def test_multi_statement_injection_rejected():
    with pytest.raises(GuardrailRejection):
        validate_and_prepare("SELECT * FROM events; DROP TABLE events")
    with pytest.raises(GuardrailRejection):
        validate_and_prepare("SELECT * FROM events; SELECT * FROM detected_anomalies")


def test_ordinary_questions_about_log_text_are_no_longer_false_positives():
    """A keyword-blocklist guardrail rejected these — 'load balancer',
    'delete', 'set' are ordinary words that appear in free-text log
    messages, exactly what an ops user searches for (Epic 5 review round 1,
    #5). The parser-based guardrail only looks at the query's real
    structure, so string-literal contents can no longer be mistaken for SQL
    keywords."""
    for sql in [
        "SELECT * FROM events WHERE message LIKE '%load balancer%'",
        "SELECT * FROM events WHERE message LIKE '%delete%'",
        "SELECT * FROM events WHERE level = 'error' AND message LIKE '%set%'",
    ]:
        result = validate_and_prepare(sql)
        assert "LIMIT" in result


def test_rejection_message_never_echoes_the_rejected_sql():
    """A rejection response must not become an oracle for probing the
    guardrail's exact pattern-matching boundaries."""
    malicious_sql = "DROP TABLE events -- secret_probe_string"
    try:
        validate_and_prepare(malicious_sql)
        pytest.fail("expected GuardrailRejection")
    except GuardrailRejection as e:
        assert "secret_probe_string" not in str(e)
        assert "DROP" not in str(e)


def test_case_insensitivity_of_select_detection():
    result = validate_and_prepare("select * from events")
    assert "LIMIT" in result
