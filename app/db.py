"""DuckDB connection helpers.

Two files, not one:
- ops.duckdb          -> events, detected_anomalies, query_log. The API's
                         text-to-SQL path may query events/detected_anomalies.
- ground_truth.duckdb -> ground_truth_anomalies. The API process never opens
                         this file, so a coerced/buggy text-to-SQL guardrail
                         can't leak it — physically unreachable, not just
                         excluded by a regex allowlist (Epic 1-2 review #2).

Two explicit read_only modes on get_connection, never a default that could
accidentally allow writes:
- read_only=True  -> the API's request-time path (text-to-SQL, /anomalies).
- read_only=False -> offline scripts only (data generation, detector runs).

read_only=True also disables DuckDB's external-access surface (COPY TO,
read_csv_auto/read_text/read_blob, httpfs). Without that, a read-only
connection still allows arbitrary local file read/write via those functions —
read_only alone only guards the database's own catalog, not the filesystem
(Epic 1-2 review #1, verified: COPY (SELECT 1) TO 'x.csv' succeeded on a
read_only=True connection before this fix).
"""
import duckdb

from app.config import settings

EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id BIGINT PRIMARY KEY,
    ts TIMESTAMP,
    service VARCHAR,
    metric_name VARCHAR,
    value DOUBLE,
    level VARCHAR,
    message VARCHAR
);
"""

_REST_OF_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detected_anomalies (
    id INTEGER PRIMARY KEY,
    service VARCHAR,
    metric_name VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    method VARCHAR,
    score DOUBLE,
    sample_event_ids BIGINT[]
);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY,
    ts TIMESTAMP,
    question VARCHAR,
    generated_sql VARCHAR,
    validation_result VARCHAR,
    row_count INTEGER,
    latency_ms INTEGER
);
"""
SCHEMA_SQL = EVENTS_TABLE_SQL + _REST_OF_SCHEMA_SQL

GROUND_TRUTH_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ground_truth_anomalies (
    id INTEGER PRIMARY KEY,
    service VARCHAR,
    metric_name VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    anomaly_type VARCHAR,
    magnitude DOUBLE
);
"""
GROUND_TRUTH_SCHEMA_SQL = GROUND_TRUTH_TABLE_SQL

# Tables the text-to-SQL layer is allowed to query (see Epic 5's guardrail).
QUERYABLE_TABLES = frozenset({"events", "detected_anomalies"})


def get_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    config = {"enable_external_access": "false"} if read_only else {}
    return duckdb.connect(settings.duckdb_path, read_only=read_only, config=config)


def get_ground_truth_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    # Only offline scripts (generator, detector eval) ever call this — never
    # the API — so it doesn't need the external-access lockdown above.
    return duckdb.connect(settings.ground_truth_duckdb_path, read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(SCHEMA_SQL)


def init_ground_truth_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(GROUND_TRUTH_SCHEMA_SQL)
