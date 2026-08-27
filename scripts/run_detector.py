"""Run both anomaly detectors over the committed events, merge overlapping
detections into one row each, and write the result to detected_anomalies.

Offline, like the generator — no live ingestion, so there's nothing to detect
in real time (see architecture doc).

Usage: python -m scripts.run_detector
"""
import pandas as pd

from app.db import DETECTED_ANOMALIES_TABLE_SQL, get_connection, init_schema
from app.detection import rolling_zscore, seasonal_residual

# A few sample intervals of slack when merging: rolling_zscore's trailing
# baseline reports onset and recovery rather than a sustained anomaly's full
# span (see its docstring), so a real event routinely fragments into several
# detection rows separated by small gaps. Without this, merge_detections
# only merges strict overlaps and one real anomaly ships as up to 9 separate
# rows (Epic 3 review round 1, #2).
_MERGE_GAP_TOLERANCE = pd.Timedelta(minutes=30)


def merge_detections(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine detections from multiple methods, merging any that overlap (or
    sit within _MERGE_GAP_TOLERANCE of each other) in time for the same
    (service, metric_name) into one row — otherwise the same real anomaly
    gets double-counted, or fragmented, in Epic 4's eval.

    Caveats for downstream readers (Epic 4/5):
    - `score` is `max()` across contributing methods' z-statistics, which
      are on different scales (a local rolling std vs. a global MAD) — not
      meaningfully comparable across methods, don't rank on it.
    - `method` is a "+"-joined string (e.g. "rolling_zscore+seasonal_residual")
      for rows both methods caught. `WHERE method = 'rolling_zscore'` misses
      those rows; use `LIKE '%rolling_zscore%'` instead.
    """
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=["service", "metric_name", "start_ts", "end_ts", "method", "score", "sample_event_ids"]
        )
    combined = pd.concat(non_empty, ignore_index=True)

    merged_rows: list[dict] = []
    for (service, metric), group in combined.groupby(["service", "metric_name"]):
        group = group.sort_values("start_ts")
        current = None
        for _, row in group.iterrows():
            if current is None:
                current = row.to_dict()
                current["method"] = {row["method"]}
                continue
            if row["start_ts"] <= current["end_ts"] + _MERGE_GAP_TOLERANCE:
                current["end_ts"] = max(current["end_ts"], row["end_ts"])
                current["score"] = max(current["score"], row["score"])
                current["method"].add(row["method"])
                current["sample_event_ids"] = sorted(set(current["sample_event_ids"]) | set(row["sample_event_ids"]))[:5]
            else:
                current["method"] = "+".join(sorted(current["method"]))
                merged_rows.append(current)
                current = row.to_dict()
                current["method"] = {row["method"]}
        if current is not None:
            current["method"] = "+".join(sorted(current["method"]))
            merged_rows.append(current)

    return pd.DataFrame(merged_rows)


def run_detector() -> pd.DataFrame:
    conn = get_connection(read_only=False)
    try:
        # Before the read: on a fresh DB (detector run before the generator's
        # own write), this makes the failure a clear "no events" empty result
        # instead of a raw CatalogException, and it's the exact ordering bug
        # Epic 1-2's round 2 regression was — keep schema setup first, always
        # (Epic 3 review round 1, nit #9).
        init_schema(conn)
        events_df = conn.execute("SELECT * FROM events ORDER BY ts").fetchdf()
        events_df["ts"] = pd.to_datetime(events_df["ts"])

        zscore_df = rolling_zscore.detect(events_df)
        seasonal_df = seasonal_residual.detect(events_df)
        merged = merge_detections(zscore_df, seasonal_df)
        merged = merged.reset_index(drop=True)
        merged.insert(0, "id", range(len(merged)))

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DROP TABLE IF EXISTS detected_anomalies")
            conn.execute(DETECTED_ANOMALIES_TABLE_SQL)
            conn.execute(
                "INSERT INTO detected_anomalies "
                "(id, service, metric_name, start_ts, end_ts, method, score, sample_event_ids) "
                "SELECT id, service, metric_name, start_ts, end_ts, method, score, sample_event_ids FROM merged"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return merged
    finally:
        conn.close()


if __name__ == "__main__":
    result = run_detector()
    print(f"detected_anomalies: {len(result)} rows")
    if not result.empty:
        print(result["method"].value_counts())
