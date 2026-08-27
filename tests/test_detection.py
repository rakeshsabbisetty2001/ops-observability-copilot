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
    # A bare 3-sigma threshold WILL occasionally flag pure noise by chance
    # (~0.27% per point under Gaussian noise) — asserting zero false
    # positives on any single seed is the wrong test. Check the rate across
    # many independent series stays in the right ballpark instead.
    n_points_total = 0
    n_flagged_points = 0
    for seed in range(20):
        df = _flat_series(n=200, seed=seed)
        result = rolling_zscore.detect(df, window=12, threshold=3.0)
        n_points_total += len(df)
        n_flagged_points += sum((row["end_ts"] - row["start_ts"]).total_seconds() / 300 + 1 for _, row in result.iterrows())
    rate = n_flagged_points / n_points_total
    assert rate < 0.02, f"false-positive rate {rate:.4f} is far above the ~0.27% expected at 3 sigma"


def test_seasonal_residual_flags_deviation_from_hourly_baseline():
    # Two clean days of an hourly pattern, then a big deviation at one hour.
    ts = pd.date_range("2026-01-01", periods=48, freq="1h")
    values = np.array([100.0 + (10.0 if h % 24 in (12, 13) else 0.0) for h in range(48)])
    df = pd.DataFrame(
        {
            "id": range(48),
            "ts": ts,
            "service": "svc",
            "metric_name": "m",
            "value": values,
            "level": "info",
            "message": "",
        }
    )
    # Break the pattern once — a spike outside the learned midday bump.
    df.loc[5, "value"] = 500.0
    result = seasonal_residual.detect(df, threshold=3.0)
    assert not result.empty
    assert (result["start_ts"] <= ts[5]).any() and (result["end_ts"] >= ts[5]).any()


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
