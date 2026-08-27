import pytest

from app.nl2sql.guardrail import DEFAULT_ROW_LIMIT, GuardrailRejection, validate_and_prepare


def test_plain_select_passes_and_gets_a_limit_injected():
    result = validate_and_prepare("SELECT * FROM events")
    assert result == f"SELECT * FROM events LIMIT {DEFAULT_ROW_LIMIT}"


def test_existing_limit_is_not_duplicated():
    result = validate_and_prepare("SELECT * FROM events LIMIT 10")
    assert result.count("LIMIT") == 1
    assert "LIMIT 10" in result


def test_trailing_semicolon_is_tolerated():
    result = validate_and_prepare("SELECT * FROM events;")
    assert ";" not in result


def test_join_across_two_allowed_tables_passes():
    result = validate_and_prepare(
        "SELECT e.* FROM events e JOIN detected_anomalies d ON e.service = d.service"
    )
    assert "LIMIT" in result


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM events; DROP TABLE events",
        "SELECT * FROM events; SELECT * FROM detected_anomalies",
        "SELECT * FROM events WHERE 1=1; --",
    ],
)
def test_multi_statement_injection_rejected(sql):
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


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
        "SELECT * FROM events WHERE service = 'a' UNION ALL DROP TABLE events",
        "SET enable_external_access=true",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL some_extension_function()",
        "VACUUM",
        "CHECKPOINT",
    ],
)
def test_disallowed_statement_type_rejected(sql):
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM ground_truth_anomalies",
        "SELECT * FROM query_log",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM events JOIN ground_truth_anomalies USING (service)",
        "SELECT * FROM pg_catalog.pg_tables",
    ],
)
def test_disallowed_table_rejected(sql):
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
        "SELECT 1",  # no FROM, no table reference
        "EXPLAIN SELECT * FROM events",
        "WITH x AS (SELECT 1) SELECT * FROM x",  # no real table reference
    ],
)
def test_malformed_or_tableless_query_rejected(sql):
    with pytest.raises(GuardrailRejection):
        validate_and_prepare(sql)


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


def test_case_insensitivity_of_keyword_checks():
    with pytest.raises(GuardrailRejection):
        validate_and_prepare("drop table events")
    with pytest.raises(GuardrailRejection):
        validate_and_prepare("SeLeCt * fRoM events; dRoP tAbLe events")


def test_select_lowercase_keyword_still_accepted():
    result = validate_and_prepare("select * from events")
    assert "LIMIT" in result
