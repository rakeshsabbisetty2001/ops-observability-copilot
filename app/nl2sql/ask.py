"""End-to-end /ask flow: question -> generate SQL -> guardrail -> execute
(read-only) -> summarize -> link overlapping detected_anomalies.

Every step is logged to query_log (audit trail + future eval fixture
material) via a SEPARATE write connection — the query-execution connection
stays read_only=True throughout; logging never shares that connection.
"""
import threading
import time

import pandas as pd

from app.db import get_connection, init_schema
from app.nl2sql.generate import generate_sql
from app.nl2sql.guardrail import GuardrailRejection, validate_and_prepare
from app.nl2sql.summarize import summarize_result

_QUERY_TIMEOUT_SECONDS = 5


def _execute_with_timeout(conn, sql: str, timeout_seconds: float) -> pd.DataFrame:
    """DuckDB has no built-in statement timeout — run the query on a worker
    thread and conn.interrupt() it from here if it overruns, so one
    pathological generated query (e.g. an accidental cross join) can't hang
    the request indefinitely."""
    outcome: dict = {}

    def _run():
        try:
            outcome["df"] = conn.execute(sql).fetchdf()
        except Exception as e:  # noqa: BLE001 - re-raised on the calling thread below
            outcome["error"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        conn.interrupt()
        worker.join(1)
        raise TimeoutError(f"query exceeded {timeout_seconds}s timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["df"]


def ask(question: str) -> dict:
    start = time.monotonic()
    generated_sql = None
    validation_result = "ok"
    row_count = None

    try:
        generated_sql = generate_sql(question)
        safe_sql = validate_and_prepare(generated_sql)
    except GuardrailRejection as e:
        validation_result = str(e)
        _log(question, generated_sql, validation_result, row_count=None, latency_ms=_elapsed_ms(start))
        return {"question": question, "sql": generated_sql, "error": validation_result, "answer": None, "row_count": None, "anomaly_ids": []}

    conn = get_connection(read_only=True)
    try:
        result_df = _execute_with_timeout(conn, safe_sql, _QUERY_TIMEOUT_SECONDS)
    except Exception as e:
        # Close BEFORE logging, not in a `finally` — `_log` opens its own
        # read_only=False connection to the same file, and DuckDB refuses to
        # open a second connection with a different config while this one is
        # still open (real bug, caught by a test: raised ConnectionException
        # instead of the intended clean error response).
        conn.close()
        validation_result = f"execution error: {type(e).__name__}"
        _log(question, generated_sql, validation_result, row_count=None, latency_ms=_elapsed_ms(start))
        return {"question": question, "sql": safe_sql, "error": "the generated query could not be executed", "answer": None, "row_count": None, "anomaly_ids": []}
    conn.close()

    row_count = len(result_df)
    rows = result_df.to_dict("records")
    answer = summarize_result(question, rows)
    anomaly_ids = _find_overlapping_anomalies(result_df)

    _log(question, generated_sql, validation_result, row_count=row_count, latency_ms=_elapsed_ms(start))

    return {
        "question": question,
        "sql": safe_sql,
        "answer": answer,
        "row_count": row_count,
        "anomaly_ids": anomaly_ids,
    }


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _find_overlapping_anomalies(result_df: pd.DataFrame) -> list[int]:
    """If the result names a service/metric and a timestamp range, look up
    detected_anomalies overlapping that range for the same series — lets the
    UI offer a drill-down link. Best-effort: a result shaped differently
    (aggregates, no service/metric columns) simply gets no links."""
    if result_df.empty or "service" not in result_df.columns or "metric_name" not in result_df.columns:
        return []
    ts_cols = [c for c in result_df.columns if pd.api.types.is_datetime64_any_dtype(result_df[c])]
    if not ts_cols:
        return []
    ts_col = ts_cols[0]

    conn = get_connection(read_only=True)
    try:
        ids: set[int] = set()
        for (service, metric), group in result_df.groupby(["service", "metric_name"]):
            lo, hi = group[ts_col].min(), group[ts_col].max()
            rows = conn.execute(
                "SELECT id FROM detected_anomalies WHERE service = ? AND metric_name = ? "
                "AND start_ts <= ? AND end_ts >= ?",
                [service, metric, hi, lo],
            ).fetchall()
            ids.update(r[0] for r in rows)
        return sorted(ids)
    finally:
        conn.close()


def _log(question: str, generated_sql: str | None, validation_result: str, row_count: int | None, latency_ms: int) -> None:
    conn = get_connection(read_only=False)
    try:
        init_schema(conn)
        next_id = conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM query_log").fetchone()[0]
        conn.execute(
            "INSERT INTO query_log (id, ts, question, generated_sql, validation_result, row_count, latency_ms) "
            "VALUES (?, now(), ?, ?, ?, ?, ?)",
            [next_id, question, generated_sql, validation_result, row_count, latency_ms],
        )
    finally:
        conn.close()
