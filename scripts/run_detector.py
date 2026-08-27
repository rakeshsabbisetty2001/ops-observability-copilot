"""Run both anomaly detectors over the committed events, merge overlapping
detections into one row each, and write the result to detected_anomalies.

Offline, like the generator — no live ingestion, so there's nothing to detect
in real time (see architecture doc).

Usage: python -m scripts.run_detector
"""
import pandas as pd

from app.db import DETECTED_ANOMALIES_TABLE_SQL, get_connection, init_schema
from app.detection import rolling_zscore, seasonal_residual


def merge_detections(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine detections from multiple methods, merging any that overlap in
    time for the same (service, metric_name) into one row — otherwise the
    same real anomaly gets double-counted in Epic 4's eval just because two
    methods both caught it."""
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
            if row["start_ts"] <= current["end_ts"]:  # overlaps the run-in-progress
                current["end_ts"] = max(current["end_ts"], row["end_ts"])
                current["score"] = max(current["score"], row["score"])
                current["method"].add(row["method"])
                current["sample_event_ids"] = list(set(current["sample_event_ids"]) | set(row["sample_event_ids"]))[:5]
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
        events_df = conn.execute("SELECT * FROM events ORDER BY ts").fetchdf()
        events_df["ts"] = pd.to_datetime(events_df["ts"])

        zscore_df = rolling_zscore.detect(events_df)
        seasonal_df = seasonal_residual.detect(events_df)
        merged = merge_detections(zscore_df, seasonal_df)
        merged = merged.reset_index(drop=True)
        merged.insert(0, "id", range(len(merged)))

        conn.execute("BEGIN TRANSACTION")
        try:
            init_schema(conn)
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
