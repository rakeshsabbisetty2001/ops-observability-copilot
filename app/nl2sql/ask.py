"""End-to-end /ask flow: question -> generate SQL -> guardrail -> execute
(read-only) -> summarize -> link overlapping detected_anomalies.

Every step is logged to query_log (audit trail + future eval fixture
material) via a single persistent connection to query_log's OWN file,
serialized with a lock — never an ad-hoc connection to ops.duckdb (DuckDB
refuses to open a read_only connection to a file that has any other
connection open with a different read_only value, so a per-call write
connection to ops.duckdb would collide with the read-only query connection
under real concurrency; reproduced live during Epic 5's review, round 1 #3).
"""
import logging
import threading
import time

import pandas as pd

_logger = logging.getLogger(__name__)

from app.db import QUERY_LOG_TABLE_SQL, get_connection, get_query_log_connection
from app.nl2sql.generate import generate_sql
from app.nl2sql.guardrail import GuardrailRejection, validate_and_prepare
from app.nl2sql.summarize import summarize_result

_QUERY_TIMEOUT_SECONDS = 5
# Independent of whatever row LIMIT the guardrail enforced — bounds what
# actually reaches the summarizer prompt, since a row count under the limit
# says nothing about payload size (Epic 5 review round 1, #6: a single
# `repeat()` call put ~800MB across 400 rows, well under a 500-row cap).
_MAX_ROWS_TO_SUMMARIZE = 50
_MAX_CELL_CHARS = 500

# A generic, fixed message for every failure surfaced to a client — never
# the specific reason, the rejected SQL, or an exception type/message, which
# would turn a response into an oracle for mapping the guardrail's exact
# boundaries or leaking internals (Epic 5 review round 1, #4: the exception
# itself already withheld this, but ask() was handing both back one frame
# later). The real reason always still goes to query_log.
_GENERIC_ERROR_MESSAGE = "the question could not be answered"

_query_log_lock = threading.Lock()
_query_log_conn = None


def _get_query_log_conn():
    global _query_log_conn
    if _query_log_conn is None:
        conn = get_query_log_connection(read_only=False)
        try:
            conn.execute(QUERY_LOG_TABLE_SQL)
        except Exception:
            # Don't leave a connection assigned to the global if schema
            # setup didn't fully complete — a partial failure here once
            # silently poisoned every later call in the process (the
            # connection object existed, so the `is None` check never
            # retried, but the table it needed was never actually created).
            conn.close()
            raise
        _query_log_conn = conn
    return _query_log_conn


class _AbandonedWorkerError(TimeoutError):
    """A query overran its timeout AND didn't die after being interrupted.
    The caller must NOT close the connection this was raised for — the
    worker thread may still be inside conn.execute() on it, and closing a
    connection out from under a still-running query deadlocks every later
    duckdb.connect() to that file for the rest of the process (Epic 5
    review round 2, #3 — reproduced directly: with the join window widened,
    6 of 10 runs hung indefinitely, confirmed via thread stack dumps to be
    new requests blocked in duckdb.connect() behind an orphaned worker still
    executing on the connection ask() had already closed). One leaked
    connection object is a far smaller cost than a process-wide hang."""


# ponytail: unbounded — each abandoned worker is a permanently-running
# thread on a shared memory_limit budget, and this only counts/logs them
# rather than capping how many can pile up. A real abandonment is rare in
# practice (interrupt() succeeds within 1-3 attempts the overwhelming
# majority of the time, measured during review), so this is enough to make
# the ceiling visible without adding real throttling machinery for a
# failure mode that hasn't happened yet — cap it if this ever serves real
# concurrent traffic (Epic 5 review round 3, #4).
_abandoned_worker_count = 0
_abandoned_worker_count_lock = threading.Lock()


def _execute_with_timeout(conn, sql: str, timeout_seconds: float) -> pd.DataFrame:
    """DuckDB has no built-in statement timeout — run the query on a worker
    thread and conn.interrupt() it from here if it overruns, so one
    pathological generated query (e.g. an accidental cross join) can't hang
    the request indefinitely. Keeps interrupting until the worker actually
    dies rather than giving up after one bounded join — a single join(1)
    that expires doesn't mean the worker won't die a moment later, and
    abandoning it early was the direct cause of the deadlock above."""
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
        # 3 attempts at 0.5s: the first attempt is what resolves it in
        # practice (measured 10/10 within iterations 1-3 during review) —
        # worst-case added latency is ~1.5s, not the ~5s a longer loop would
        # add on top of the timeout itself (Epic 5 review round 3, nit #6:
        # _QUERY_TIMEOUT_SECONDS should still roughly describe the
        # user-visible ceiling).
        for _ in range(3):
            conn.interrupt()
            worker.join(0.5)
            if not worker.is_alive():
                raise TimeoutError(f"query exceeded {timeout_seconds}s timeout")
        with _abandoned_worker_count_lock:
            global _abandoned_worker_count
            _abandoned_worker_count += 1
            count = _abandoned_worker_count
        _logger.warning("query timed out and could not be interrupted (abandoned workers so far: %d)", count)
        raise _AbandonedWorkerError(f"query exceeded {timeout_seconds}s timeout and could not be interrupted")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["df"]


def ask(question: str) -> dict:
    """One log record per call, written in `finally`, whatever happened —
    this used to be logged separately on each failure branch, which meant a
    failure AFTER execution (summarize_result is a live network call:
    timeouts/429s/529s are routine; so is _find_overlapping_anomalies
    hitting a DB error) skipped the audit record entirely for a request that
    had actually run real SQL (Epic 5 review round 1, #9)."""
    start = time.monotonic()
    generated_sql: str | None = None
    validation_result = "did not complete"
    row_count: int | None = None

    try:
        generated_sql = generate_sql(question)
        safe_sql = validate_and_prepare(generated_sql)
        validation_result = "ok"

        conn = get_connection(read_only=True)
        try:
            result_df = _execute_with_timeout(conn, safe_sql, _QUERY_TIMEOUT_SECONDS)
        except _AbandonedWorkerError:
            raise  # do NOT close conn here — see _AbandonedWorkerError's docstring
        except BaseException:
            # Every OTHER failure (a plain TimeoutError once the worker did
            # die, or the worker's own re-raised exception — a BinderException
            # from a hallucinated column, OutOfMemoryException, etc.) skipped
            # this close entirely under the old except/else split, leaving
            # release to depend on CPython refcounting rather than an
            # explicit call (Epic 5 review round 3, #2 — confirmed via an
            # instrumented count: 0 explicit closes on this exact path).
            conn.close()
            raise
        else:
            conn.close()

        row_count = len(result_df)
        rows = _cap_rows_for_summary(result_df)
        answer = summarize_result(question, rows)
        anomaly_ids = _find_overlapping_anomalies(result_df)

        return {
            "question": question,
            "sql": safe_sql,
            "answer": answer,
            "row_count": row_count,
            "anomaly_ids": anomaly_ids,
            "error": None,
        }
    except GuardrailRejection as e:
        validation_result = str(e)
        return {"question": question, "sql": None, "answer": None, "row_count": None, "anomaly_ids": [], "error": _GENERIC_ERROR_MESSAGE}
    except Exception as e:
        # The real reason (str(e)), not just the exception type name — round
        # 1's fix was described as logging "the real failure reason" but
        # only did that for guardrail rejections; every other failure logged
        # a bare type name with the message discarded (Epic 5 review round
        # 2, #6). query_log is the audit trail; the type alone isn't enough
        # to debug a live incident from.
        validation_result = f"error: {type(e).__name__}: {e}"[:2000]
        return {"question": question, "sql": None, "answer": None, "row_count": None, "anomaly_ids": [], "error": _GENERIC_ERROR_MESSAGE}
    finally:
        # Never let a logging failure replace the response above — the audit
        # write is the least important thing in this function to fail the
        # request on (Epic 5 review round 2, #7).
        try:
            _log(question, generated_sql, validation_result, row_count, _elapsed_ms(start))
        except Exception:
            pass


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _cap_rows_for_summary(result_df: pd.DataFrame) -> list[dict]:
    capped = result_df.head(_MAX_ROWS_TO_SUMMARIZE)
    rows = capped.to_dict("records")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
                row[key] = value[:_MAX_CELL_CHARS] + "...(truncated)"
    return rows


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
    with _query_log_lock:
        conn = _get_query_log_conn()
        conn.execute(
            "INSERT INTO query_log (ts, question, generated_sql, validation_result, row_count, latency_ms) "
            "VALUES (now(), ?, ?, ?, ?, ?)",
            [question, generated_sql, validation_result, row_count, latency_ms],
        )
