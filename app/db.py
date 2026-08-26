"""DuckDB connection helper.

Two explicit modes, never a default that could accidentally allow writes:
- read_only=True  -> the API's request-time path (text-to-SQL, /anomalies).
- read_only=False -> offline scripts only (data generation, detector runs).
"""
import duckdb

from app.config import settings

# One shared schema definition so every script/writer stays in sync — the
# tables themselves are the contract (see architecture doc for field meanings).
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id BIGINT PRIMARY KEY,
    ts TIMESTAMP,
    service VARCHAR,
    metric_name VARCHAR,
    value DOUBLE,
    level VARCHAR,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS ground_truth_anomalies (
    id INTEGER PRIMARY KEY,
    service VARCHAR,
    metric_name VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    anomaly_type VARCHAR,
    magnitude DOUBLE
);

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

# Tables the text-to-SQL layer is allowed to query. ground_truth_anomalies and
# query_log are deliberately excluded — see Epic 5's guardrail.
QUERYABLE_TABLES = frozenset({"events", "detected_anomalies"})


def get_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(settings.duckdb_path, read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(SCHEMA_SQL)
