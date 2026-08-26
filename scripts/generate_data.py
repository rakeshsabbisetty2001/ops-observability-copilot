"""Generate synthetic ops logs/metrics with known injected ground-truth anomalies.

The detector (Epic 3) never sees ground_truth_anomalies — it's written here,
before detection ever runs, specifically so the eval (Epic 4) has an honest
answer key that wasn't derived from the thing being graded. It also lives in
a separate DuckDB file the API never opens (see app/db.py) — the answer key
is physically unreachable from the text-to-SQL path, not just excluded by a
guardrail regex.

Usage: python -m scripts.generate_data [--seed 42] [--days 14] [--interval-minutes 5]
"""
import argparse
import math
import random
from datetime import timedelta

import numpy as np
import pandas as pd

from app.db import (
    EVENTS_TABLE_SQL,
    GROUND_TRUTH_TABLE_SQL,
    get_connection,
    get_ground_truth_connection,
    init_schema,
)

SERVICES = ["checkout-api", "payments-worker", "auth-service"]
METRICS = {
    # baseline: normal mean. std: normal noise. seasonal_amp: daily swing (0 = none).
    # min/max: physically valid range — real-world quantities can't go negative,
    # rates/percentages are capped (Epic 1-2 review #6: uncapped noise produced
    # negative error rates in the committed corpus).
    "latency_ms": {"baseline": 120.0, "std": 15.0, "seasonal_amp": 30.0, "min": 0.0, "max": math.inf},
    "error_rate": {"baseline": 0.3, "std": 0.1, "seasonal_amp": 0.0, "min": 0.0, "max": 1.0},
    "cpu_pct": {"baseline": 35.0, "std": 5.0, "seasonal_amp": 10.0, "min": 0.0, "max": 100.0},
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
    # Fitting n non-overlapping windows of up to max_len needs roughly n * max_len
    # points, not just max_len (Epic 1-2 review round 2, #4 — the single-window
    # bound let e.g. --days 1 --interval-minutes 30 through, then crashed with
    # an opaque "only placed 2/3 windows" further down).
    if n_points <= n * max_len:
        raise ValueError(
            f"n_points ({n_points}) too small to fit {n} non-overlapping windows of up to "
            f"{max_len} points each — increase --days or decrease --interval-minutes"
        )
    windows: list[tuple[int, int]] = []
    attempts = 0
    while len(windows) < n and attempts < 200:
        attempts += 1
        length = rng.randint(min_len, max_len)
        start_idx = rng.randint(0, n_points - length)  # end_idx can reach n_points - 1 inclusive
        end_idx = start_idx + length
        if any(not (end_idx < w[0] or start_idx > w[1]) for w in windows):
            continue  # overlaps an existing window for this (service, metric) — retry
        windows.append((start_idx, end_idx))
    assert len(windows) == n, f"only placed {len(windows)}/{n} non-overlapping windows after 200 attempts"
    return windows


def generate(seed: int = 42, days: int = 14, interval_minutes: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)  # not the legacy global RNG — no side effect on other callers

    n_points = int(days * 24 * 60 / interval_minutes)
    timestamps = [START_TS + timedelta(minutes=interval_minutes * i) for i in range(n_points)]

    event_rows, gt_rows = [], []
    event_id = 0
    gt_id = 0

    for service in SERVICES:
        for metric_name, cfg in METRICS.items():
            baseline, std, amp = cfg["baseline"], cfg["std"], cfg["seasonal_amp"]
            vmin, vmax = cfg["min"], cfg["max"]

            base = np.array(
                [baseline + _seasonal_component(ts, amp) + rng_np.normal(0, std) for ts in timestamps]
            )
            values = base.copy()

            n_anomalies = rng.randint(1, 3)
            windows = _pick_anomaly_windows(rng, n_points, n_anomalies, min_len=6, max_len=36)

            anomaly_specs = []  # (start_idx, end_idx, anomaly_type, attempted_effect)
            for start_idx, end_idx in windows:
                anomaly_type = rng.choice(ANOMALY_TYPES)
                if anomaly_type == "spike":
                    attempted = rng.uniform(3.0, 6.0) * std
                    values[start_idx:end_idx] += attempted
                elif anomaly_type == "dip":
                    # Cap so the dip can't crush the window flat against the
                    # floor (baseline - attempted must clear vmin by >=1 std) —
                    # error_rate's baseline (0.3) is only ~3 std above its own
                    # floor (0), so an uncapped 3-6 std dip always bottomed out
                    # at a constant-zero line (Epic 1-2 review round 2, #2).
                    max_dip = max(baseline - vmin - std, std)
                    attempted = min(rng.uniform(3.0, 6.0) * std, max_dip)
                    values[start_idx:end_idx] -= attempted
                else:  # sustained_drift
                    # A ramp's *mean* effect over its window is only attempted/2, so
                    # needs a higher floor than spike/dip to stay reliably detectable
                    # (verified by fuzzing generate() across 200 seeds).
                    attempted = rng.uniform(3.0, 5.0) * std
                    values[start_idx:end_idx] += np.linspace(0, attempted, end_idx - start_idx)
                anomaly_specs.append((start_idx, end_idx, anomaly_type, attempted))

            values = np.clip(values, vmin, vmax)

            for start_idx, end_idx, anomaly_type, attempted in anomaly_specs:
                window_base_mean = base[start_idx:end_idx].mean()
                realized = float(values[start_idx:end_idx].mean() - window_base_mean)

                # Independent sanity check: the *attempted* effect (known from the
                # rng draw, not measured from the array) vs what actually landed,
                # accounting for clipping analytically. Catches a silently no-op'd
                # injection (realized ~= 0 while theoretical is clearly not) —
                # comparing against the *recorded* magnitude instead would be
                # self-referential, since both would read ~0 (Epic 1-2 review #4).
                if anomaly_type == "spike":
                    theoretical = min(attempted, vmax - window_base_mean) if math.isfinite(vmax) else attempted
                elif anomaly_type == "dip":
                    theoretical = -min(attempted, window_base_mean - vmin)
                else:  # sustained_drift: unclipped mean effect is attempted/2
                    # Same clipping-awareness as spike, halved — a ramp that
                    # would push the window mean past vmax gets bounded like a
                    # spike does. Doesn't currently bind at these parameters,
                    # but leaving it exact avoids a landmine if the drift
                    # multiplier or a bound ever changes (round 2, nit #6).
                    theoretical = (min(attempted, vmax - window_base_mean) if math.isfinite(vmax) else attempted) / 2
                tolerance = 3.0 * std / math.sqrt(end_idx - start_idx)
                assert abs(realized - theoretical) < tolerance, (
                    f"{anomaly_type} for {service}/{metric_name}: realized shift {realized:.4f} doesn't "
                    f"match theoretical {theoretical:.4f} (±{tolerance:.4f}) — injection likely no-op'd"
                )

                gt_rows.append(
                    {
                        "id": gt_id,
                        "service": service,
                        "metric_name": metric_name,
                        "start_ts": timestamps[start_idx],
                        "end_ts": timestamps[end_idx - 1],
                        "anomaly_type": anomaly_type,
                        "magnitude": abs(realized),  # the REAL post-clip effect, not the attempted target
                    }
                )
                gt_id += 1

            for i, ts in enumerate(timestamps):
                is_anomalous = any(s <= i < e for s, e, _, _ in anomaly_specs)
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
    """Real assertions, not comments — catches off-by-one windows and
    corruption introduced between the in-memory arrays and the final
    DataFrame (e.g. a column-order mismatch, a rounding artifact, a bad id).
    This is NOT the no-op-injection detector — that's the exact (non-
    statistical) check inside generate()'s per-window loop, which compares
    the realized effect against the independently-known rng-drawn target.
    This check instead cross-validates the *recorded* magnitude against an
    independently-recomputed observed shift; at a no-op both numbers would
    read ~0 and agree, so on its own it has poor power against that failure
    mode (round 2 review, #3)."""
    assert not events_df.empty and not gt_df.empty, "generation produced no rows"

    ts_min, ts_max = events_df["ts"].min(), events_df["ts"].max()
    assert (gt_df["start_ts"] >= ts_min).all() and (gt_df["end_ts"] <= ts_max).all(), (
        "ground-truth window falls outside the generated event range"
    )

    residual = _residuals(events_df)

    for (service, metric), group in gt_df.groupby(["service", "metric_name"]):
        windows = sorted(zip(group["start_ts"], group["end_ts"]))
        for (_, e1), (s2, _) in zip(windows, windows[1:]):
            assert e1 < s2, f"overlapping ground-truth windows for {service}/{metric}"

        all_mask = (events_df["service"] == service) & (events_df["metric_name"] == metric)
        in_any_window = pd.Series(False, index=events_df.index)
        for _, w in group.iterrows():
            in_any_window |= (events_df["ts"] >= w["start_ts"]) & (events_df["ts"] <= w["end_ts"])
        clean = residual.loc[all_mask & ~in_any_window]
        clean_mean, clean_std = clean.mean(), clean.std()
        if clean_std <= 0:
            clean_std = METRICS[metric]["std"]

        n_clean = len(clean)
        for _, row in group.iterrows():
            win_mask = all_mask & (events_df["ts"] >= row["start_ts"]) & (events_df["ts"] <= row["end_ts"])
            n_win = int(win_mask.sum())
            win_mean = residual.loc[win_mask].mean()
            observed = win_mean - clean_mean
            expected = row["magnitude"] if row["anomaly_type"] in ("spike", "sustained_drift") else -row["magnitude"]
            # observed's own noise has two independent sources: this window's
            # actual (uncancelled) noise realization, and clean_mean's own
            # sampling error — unlike the in-generate() check above, this one
            # compares against the *aggregate* baseline, not this window's own
            # pre-injection values, so both terms belong in the tolerance.
            # 6 sigma (P ~= 2e-9 per window, ~3e-8 per run at 17 windows) — 4
            # sigma flaked ~1% of production-scale runs (seeds 21/49/74) on a
            # plain statistical tail, not clipping as first suspected (round 2
            # review, #3 — re-derived and confirmed the formula itself is
            # correct empirically, only the threshold was too tight).
            standard_error = clean_std * np.sqrt(1.0 / n_win + 1.0 / n_clean)
            assert abs(observed - expected) < 6.0 * standard_error, (
                f"recorded magnitude for {row['anomaly_type']} on {service}/{metric} doesn't match what's "
                f"actually in the persisted events (observed {observed:.4f} vs expected {expected:.4f})"
            )


def write_to_db(events_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    # DROP + CREATE + INSERT in one transaction, not DELETE + INSERT: DuckDB's
    # primary-key index does not see a same-transaction DELETE when checking a
    # same-key INSERT right after (a documented DuckDB index limitation,
    # reproduced directly while building this) — a fresh table sidesteps it
    # and gives the same "old data or new data, never a mix" guarantee.
    conn = get_connection(read_only=False)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            # detected_anomalies + query_log, IF NOT EXISTS — a targeted
            # DROP+CREATE of just `events` below must not leave the other two
            # tables uncreated (Epic 1-2 review round 2, #1: this call was
            # dropped by the round-1 rewrite and silently deleted both tables
            # from the shipped corpus).
            init_schema(conn)
            conn.execute("DROP TABLE IF EXISTS events")
            conn.execute(EVENTS_TABLE_SQL)
            conn.execute(
                "INSERT INTO events (id, ts, service, metric_name, value, level, message) "
                "SELECT id, ts, service, metric_name, value, level, message FROM events_df"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    gt_conn = get_ground_truth_connection(read_only=False)
    try:
        gt_conn.execute("BEGIN TRANSACTION")
        try:
            gt_conn.execute("DROP TABLE IF EXISTS ground_truth_anomalies")
            gt_conn.execute(GROUND_TRUTH_TABLE_SQL)
            gt_conn.execute(
                "INSERT INTO ground_truth_anomalies (id, service, metric_name, start_ts, end_ts, anomaly_type, magnitude) "
                "SELECT id, service, metric_name, start_ts, end_ts, anomaly_type, magnitude FROM gt_df"
            )
            gt_conn.execute("COMMIT")
        except Exception:
            gt_conn.execute("ROLLBACK")
            raise
    finally:
        gt_conn.close()


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
