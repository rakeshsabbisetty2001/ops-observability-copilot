"""DuckDB connection helpers.

Three files, not one:
- ops.duckdb          -> events, detected_anomalies. The API's text-to-SQL
                         path may query both, always read_only=True.
- ground_truth.duckdb -> ground_truth_anomalies. The API process never opens
                         this file, so a coerced/buggy text-to-SQL guardrail
                         can't leak it — physically unreachable, not just
                         excluded by a regex allowlist (Epic 1-2 review #2).
- query_log.duckdb    -> query_log, the per-request audit trail. Separate
                         from ops.duckdb because DuckDB refuses to open a
                         read_only connection to a file that has ANY other
                         connection open with a different read_only value —
                         verified directly, matching configs does not fix
                         it. The API's ops.duckdb connections must stay
                         read_only=True at all times (that's the whole
                         point), and query_log needs to write on every
                         request, so the two cannot share a file (Epic 5
                         review round 1, #3).

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

DETECTED_ANOMALIES_TABLE_SQL = """
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
"""
SCHEMA_SQL = EVENTS_TABLE_SQL + DETECTED_ANOMALIES_TABLE_SQL

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
# ground_truth.duckdb holds exactly this one table, so write_to_db's own
# DROP+CREATE of it is the only schema setup that file ever needs — no
# separate init function (Epic 1-2 review round 2, nit #5: the previous
# init_ground_truth_schema had no callers and was dead code).

# query_log accumulates across the process's lifetime (never rebuilt like
# events/ground_truth), so its id comes from a real DuckDB SEQUENCE rather
# than a hand-rolled SELECT MAX(id)+1 — the latter is a read-then-write race
# under FastAPI's default concurrent request handling (Epic 5 review round
# 1, #3: reproduced live, concurrent requests got duplicate-PK failures).
QUERY_LOG_TABLE_SQL = """
CREATE SEQUENCE IF NOT EXISTS query_log_id_seq START 1;
CREATE TABLE IF NOT EXISTS query_log (
    id BIGINT PRIMARY KEY DEFAULT nextval('query_log_id_seq'),
    ts TIMESTAMP,
    question VARCHAR,
    generated_sql VARCHAR,
    validation_result VARCHAR,
    row_count INTEGER,
    latency_ms INTEGER
);
"""

# Tables the text-to-SQL layer is allowed to query (see Epic 5's guardrail).
# query_log is deliberately absent — it lives in its own file the guardrail
# path never opens at all, the same physical-isolation pattern as ground truth.
QUERYABLE_TABLES = frozenset({"events", "detected_anomalies"})


def get_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    # memory_limit bounds a single query's own memory use, independent of
    # any row LIMIT: a row count under the limit says nothing about payload
    # size — `SELECT repeat(message, 2000000) FROM events LIMIT 400`
    # allocated enough to hit an OutOfMemoryException (once, an unhandled
    # native crash) with the row cap fully satisfied (Epic 5 review round 1,
    # #6). Only set on the API's read-only path; offline scripts write the
    # full corpus and shouldn't be constrained by a request-sized budget.
    config = {"enable_external_access": "false", "memory_limit": "512MB"} if read_only else {}
    return duckdb.connect(settings.duckdb_path, read_only=read_only, config=config)


def get_ground_truth_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    # Only offline scripts (generator, detector eval) ever call this — never
    # the API — so it doesn't need the external-access lockdown above.
    return duckdb.connect(settings.ground_truth_duckdb_path, read_only=read_only)


def get_query_log_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(settings.query_log_duckdb_path, read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(SCHEMA_SQL)
