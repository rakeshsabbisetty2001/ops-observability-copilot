import numpy as np
import pandas as pd
import pytest

from app.detection import rolling_zscore, seasonal_residual
from scripts.run_detector import merge_detections


def _flat_series(n=100, baseline=100.0, std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    values = baseline + rng.normal(0, std, n)
    return pd.DataFrame(
        {
            "id": range(n),
            "ts": ts,
            "service": "svc",
            "metric_name": "m",
            "value": values,
            "level": "info",
            "message": "",
        }
    )


def test_rolling_zscore_flags_an_obvious_spike():
    df = _flat_series()
    df.loc[50:55, "value"] += 50  # way outside 1-std noise
    result = rolling_zscore.detect(df, window=12, threshold=3.0)
    assert not result.empty
    # A 3-sigma threshold can also catch an occasional false positive from
    # plain noise elsewhere in the series (expected — see the false-positive
    # rate test below) — check that SOME row covers the real spike, not that
    # it's the only or first one.
    overlaps_spike = (result["start_ts"] <= df.loc[55, "ts"]) & (result["end_ts"] >= df.loc[50, "ts"])
    assert overlaps_spike.any()


def test_rolling_zscore_false_positive_rate_is_reasonable_on_clean_noise():
    # A trailing sample std over `window` prior points makes this a
    # Student-t statistic, not normal — the real single-point false-positive
    # rate at threshold=3.0 is ~1.4% (measured), not the ~0.27% a normal-
    # theory reading of "3 sigma" would suggest (Epic 3 review round 1, #1 —
    # that earlier, wrong assumption is exactly why this bound used to be
    # vacuous). The min_run=2 persistence requirement is what actually buys
    # back precision: it drops the *window*-level rate to ~0.01%, since two
    # consecutive noise points both crossing the threshold is rare even
    # though one point alone isn't.
    n_points_total = 0
    n_flagged_points = 0
    for seed in range(20):
        df = _flat_series(n=200, seed=seed)
        result = rolling_zscore.detect(df, window=12, threshold=3.0)
        n_points_total += len(df)
        n_flagged_points += sum((row["end_ts"] - row["start_ts"]).total_seconds() / 300 + 1 for _, row in result.iterrows())
    rate = n_flagged_points / n_points_total
    assert rate < 0.002, f"false-positive rate {rate:.4f} is far above the ~0.01% expected with min_run=2"


def test_seasonal_residual_flags_deviation_from_hourly_baseline():
    # 7 clean days of an hourly pattern (7 samples per hour-of-day bucket) —
    # enough for the median to have a real breakdown point. At only 2
    # samples/bucket, a median IS the mean (no robustness at all); this needs
    # a majority-clean bucket for the fix in #3 to mean anything.
    # Real noise, not a perfectly deterministic signal: with zero noise, more
    # than half the residuals are an exact 0.0, so the MAD itself degenerates
    # to 0 (median of mostly-zero absolute deviations) and nothing can ever
    # be flagged — a real telemetry series never behaves that way (see
    # generate_data.py, which always adds Gaussian noise), so a noiseless
    # fixture was testing a case the real corpus can't hit.
    rng = np.random.default_rng(0)
    n_days = 7
    ts = pd.date_range("2026-01-01", periods=24 * n_days, freq="1h")
    values = np.array([100.0 + (10.0 if h % 24 in (12, 13) else 0.0) for h in range(24 * n_days)]) + rng.normal(
        0, 0.5, 24 * n_days
    )
    df = pd.DataFrame(
        {
            "id": range(24 * n_days),
            "ts": ts,
            "service": "svc",
            "metric_name": "m",
            "value": values,
            "level": "info",
            "message": "",
        }
    )
    # Break the pattern once, on a single day — a 2-point spike outside the
    # learned midday bump (2 points, not 1, to clear flags_to_windows'
    # min_run=2 persistence requirement).
    anomaly_idx = 24 * 3 + 5  # day 3, hour 5
    df.loc[anomaly_idx : anomaly_idx + 1, "value"] = 500.0
    result = seasonal_residual.detect(df, threshold=3.0)
    # With a mean/std baseline this used to return an extra window — a
    # mirror false positive on a clean day at the same hour-of-day, because
    # the contaminated mean shifted every point's residual at that hour, not
    # just the anomalous one. With 6 of 7 samples/bucket clean, the median
    # stays put and only the real anomaly should survive (Epic 3 review
    # round 1, #3 — this test used to pass on `.any()` alone, which hid it).
    assert len(result) == 1
    row = result.iloc[0]
    assert row["start_ts"] <= ts[anomaly_idx] <= row["end_ts"]


def test_merge_detections_combines_overlapping_windows_from_both_methods():
    common = dict(service="svc", metric_name="m")
    a = pd.DataFrame(
        [{**common, "start_ts": pd.Timestamp("2026-01-01 00:00"), "end_ts": pd.Timestamp("2026-01-01 01:00"),
          "method": "rolling_zscore", "score": 3.5, "sample_event_ids": [1, 2]}]
    )
    b = pd.DataFrame(
        [{**common, "start_ts": pd.Timestamp("2026-01-01 00:30"), "end_ts": pd.Timestamp("2026-01-01 02:00"),
          "method": "seasonal_residual", "score": 4.0, "sample_event_ids": [3, 4]}]
    )
    merged = merge_detections(a, b)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["end_ts"] == pd.Timestamp("2026-01-01 02:00")
    assert row["score"] == 4.0
    assert set(row["method"].split("+")) == {"rolling_zscore", "seasonal_residual"}
    # sorted union, not an arbitrary hash-order slice (Epic 3 review round 1, #5)
    assert row["sample_event_ids"] == [1, 2, 3, 4]


def test_merge_detections_bridges_a_small_gap():
    # rolling_zscore reports onset and recovery separately for a sustained
    # anomaly, not its full span (see its docstring) — fragments separated
    # by small gaps must still merge into one row (Epic 3 review round 1, #2).
    common = dict(service="svc", metric_name="m", method="rolling_zscore", score=3.0, sample_event_ids=[1])
    a = pd.DataFrame([{**common, "start_ts": pd.Timestamp("2026-01-01 00:00"), "end_ts": pd.Timestamp("2026-01-01 00:10")}])
    b = pd.DataFrame([{**common, "start_ts": pd.Timestamp("2026-01-01 00:20"), "end_ts": pd.Timestamp("2026-01-01 00:30")}])
    merged = merge_detections(a, b)
    assert len(merged) == 1
    assert merged.iloc[0]["end_ts"] == pd.Timestamp("2026-01-01 00:30")


def test_merge_detections_keeps_non_overlapping_windows_separate():
    common = dict(service="svc", metric_name="m", method="rolling_zscore", score=3.0, sample_event_ids=[1])
    a = pd.DataFrame([{**common, "start_ts": pd.Timestamp("2026-01-01 00:00"), "end_ts": pd.Timestamp("2026-01-01 00:10")}])
    b = pd.DataFrame([{**common, "start_ts": pd.Timestamp("2026-01-02 00:00"), "end_ts": pd.Timestamp("2026-01-02 00:10")}])
    merged = merge_detections(a, b)
    assert len(merged) == 2


def test_merge_detections_empty_input():
    empty = pd.DataFrame(columns=["service", "metric_name", "start_ts", "end_ts", "method", "score", "sample_event_ids"])
    result = merge_detections(empty, empty)
    assert result.empty
