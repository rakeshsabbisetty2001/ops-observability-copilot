"""Generate synthetic ops logs/metrics with known injected ground-truth anomalies.

The detector (Epic 3) never sees ground_truth_anomalies — it's written here,
before detection ever runs, specifically so the eval (Epic 4) has an honest
answer key that wasn't derived from the thing being graded.

Usage: python -m scripts.generate_data [--seed 42] [--days 14] [--interval-minutes 5]
"""
import argparse
import random
from datetime import timedelta

import numpy as np
import pandas as pd

from app.db import get_connection, init_schema

SERVICES = ["checkout-api", "payments-worker", "auth-service"]
METRICS = {
    # baseline: normal mean. std: normal noise. seasonal_amp: daily swing (0 = none).
    "latency_ms": {"baseline": 120.0, "std": 15.0, "seasonal_amp": 30.0},
    "error_rate": {"baseline": 0.3, "std": 0.1, "seasonal_amp": 0.0},
    "cpu_pct": {"baseline": 35.0, "std": 5.0, "seasonal_amp": 10.0},
}
ANOMALY_TYPES = ["spike", "dip", "sustained_drift"]
START_TS = pd.Timestamp("2026-08-01 00:00:00")


def _seasonal_component(ts: pd.Timestamp, amplitude: float) -> float:
    """Daily seasonality peaking mid-afternoon, zero for metrics with no seasonal_amp."""
    if amplitude == 0.0:
        return 0.0
    hour_frac = (ts.hour + ts.minute / 60) / 24
    return amplitude * max(0.0, np.sin((hour_frac - 0.25) * 2 * np.pi))


def _pick_anomaly_windows(rng: random.Random, n_points: int, n: int, min_len: int, max_len: int):
    """Pick n non-overlapping (start_idx, end_idx) windows into a series of n_points."""
    windows: list[tuple[int, int]] = []
    attempts = 0
    while len(windows) < n and attempts < 200:
        attempts += 1
        length = rng.randint(min_len, max_len)
        start_idx = rng.randint(0, n_points - length - 1)
        end_idx = start_idx + length
        if any(not (end_idx < w[0] or start_idx > w[1]) for w in windows):
            continue  # overlaps an existing window for this (service, metric) — retry
        windows.append((start_idx, end_idx))
    return windows


def generate(seed: int = 42, days: int = 14, interval_minutes: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    np.random.seed(seed)

    n_points = int(days * 24 * 60 / interval_minutes)
    timestamps = [START_TS + timedelta(minutes=interval_minutes * i) for i in range(n_points)]

    event_rows, gt_rows = [], []
    event_id = 0
    gt_id = 0

    for service in SERVICES:
        for metric_name, cfg in METRICS.items():
            baseline, std, amp = cfg["baseline"], cfg["std"], cfg["seasonal_amp"]

            values = np.array(
                [baseline + _seasonal_component(ts, amp) + np.random.normal(0, std) for ts in timestamps]
            )

            n_anomalies = rng.randint(1, 3)
            windows = _pick_anomaly_windows(rng, n_points, n_anomalies, min_len=6, max_len=36)

            for start_idx, end_idx in windows:
                anomaly_type = rng.choice(ANOMALY_TYPES)
                if anomaly_type == "spike":
                    magnitude = rng.uniform(3.0, 6.0) * std
                    values[start_idx:end_idx] += magnitude
                elif anomaly_type == "dip":
                    magnitude = rng.uniform(3.0, 6.0) * std
                    values[start_idx:end_idx] = np.maximum(values[start_idx:end_idx] - magnitude, 0.0)
                else:  # sustained_drift
                    # A ramp's *mean* effect over its window is only magnitude/2, so
                    # needs a higher floor than spike/dip to stay reliably detectable
                    # (verified by fuzzing generate() across seeds/window lengths).
                    magnitude = rng.uniform(3.0, 5.0) * std
                    values[start_idx:end_idx] += np.linspace(0, magnitude, end_idx - start_idx)

                gt_rows.append(
                    {
                        "id": gt_id,
                        "service": service,
                        "metric_name": metric_name,
                        "start_ts": timestamps[start_idx],
                        "end_ts": timestamps[end_idx - 1],
                        "anomaly_type": anomaly_type,
                        "magnitude": float(magnitude),
                    }
                )
                gt_id += 1

            for i, ts in enumerate(timestamps):
                is_anomalous = any(s <= i < e for s, e in windows)
                roll = rng.random()
                if is_anomalous:
                    level = "error" if roll < 0.4 else ("warn" if roll < 0.8 else "info")
                else:
                    level = "error" if roll < 0.01 else ("warn" if roll < 0.05 else "info")
                event_rows.append(
                    {
                        "id": event_id,
                        "ts": ts,
                        "service": service,
                        "metric_name": metric_name,
                        "value": round(float(values[i]), 3),
                        "level": level,
                        "message": f"{metric_name}={values[i]:.2f} on {service}",
                    }
                )
                event_id += 1

    events_df = pd.DataFrame(event_rows)
    gt_df = pd.DataFrame(gt_rows)
    _validate(events_df, gt_df)
    return events_df, gt_df


def _residuals(events_df: pd.DataFrame) -> pd.Series:
    """value minus this metric's own baseline+seasonal formula — isolates noise
    and any injected anomaly effect, with daily seasonality removed."""
    hour_frac = (events_df["ts"].dt.hour + events_df["ts"].dt.minute / 60) / 24
    amp = events_df["metric_name"].map(lambda m: METRICS[m]["seasonal_amp"])
    baseline = events_df["metric_name"].map(lambda m: METRICS[m]["baseline"])
    seasonal = amp * np.maximum(0.0, np.sin((hour_frac - 0.25) * 2 * np.pi))
    return events_df["value"] - baseline - seasonal


def _validate(events_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    """Real assertions, not comments — catches off-by-one windows or a no-op injection."""
    ts_min, ts_max = events_df["ts"].min(), events_df["ts"].max()
    assert (gt_df["start_ts"] >= ts_min).all() and (gt_df["end_ts"] <= ts_max).all(), (
        "ground-truth window falls outside the generated event range"
    )

    residual = _residuals(events_df)

    for (service, metric), group in gt_df.groupby(["service", "metric_name"]):
        windows = sorted(zip(group["start_ts"], group["end_ts"]))
        for (_, e1), (s2, _) in zip(windows, windows[1:]):
            assert e1 < s2, f"overlapping ground-truth windows for {service}/{metric}"

        # An injection that didn't actually move the data would silently make the
        # detector eval meaningless. Compare each window's residual MEAN (value
        # with baseline+seasonal already subtracted, so no daily-cycle confound)
        # against a clean residual baseline (this metric's own points minus every
        # one of its anomaly windows) using a standard-error-scaled threshold —
        # a plain fixed multiple of std would spuriously fail short windows
        # where a real-but-small ramp is comparable in size to per-point noise
        # (the mean over the window converges even when the median doesn't).
        all_mask = (events_df["service"] == service) & (events_df["metric_name"] == metric)
        in_any_window = pd.Series(False, index=events_df.index)
        for _, w in group.iterrows():
            in_any_window |= (events_df["ts"] >= w["start_ts"]) & (events_df["ts"] <= w["end_ts"])
        clean = residual.loc[all_mask & ~in_any_window]
        clean_mean, clean_std = clean.mean(), clean.std()
        if clean_std <= 0:
            clean_std = METRICS[metric]["std"]

        for _, row in group.iterrows():
            win_mask = all_mask & (events_df["ts"] >= row["start_ts"]) & (events_df["ts"] <= row["end_ts"])
            n_win = int(win_mask.sum())
            win_mean = residual.loc[win_mask].mean()
            standard_error = clean_std / np.sqrt(n_win)
            assert abs(win_mean - clean_mean) > 2.0 * standard_error, (
                f"injected {row['anomaly_type']} for {service}/{metric} did not measurably move the data"
            )


def write_to_db(events_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    conn = get_connection(read_only=False)
    try:
        init_schema(conn)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM ground_truth_anomalies")
        conn.execute("INSERT INTO events SELECT * FROM events_df")
        conn.execute("INSERT INTO ground_truth_anomalies SELECT * FROM gt_df")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--interval-minutes", type=int, default=5)
    args = parser.parse_args()

    events_df, gt_df = generate(seed=args.seed, days=args.days, interval_minutes=args.interval_minutes)
    write_to_db(events_df, gt_df)
    print(f"events: {len(events_df)} rows, ground_truth_anomalies: {len(gt_df)} rows")
    print(gt_df["anomaly_type"].value_counts())
