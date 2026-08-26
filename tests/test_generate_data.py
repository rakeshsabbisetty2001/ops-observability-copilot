import pandas as pd
import pytest

from scripts.generate_data import METRICS, SERVICES, _validate, generate


def test_generate_shape():
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    expected_points = int(2 * 24 * 60 / 15) * len(SERVICES) * len(METRICS)
    assert len(events_df) == expected_points
    assert set(events_df["service"]) == set(SERVICES)
    assert set(events_df["metric_name"]) == set(METRICS)
    assert not gt_df.empty


def test_ground_truth_within_range_and_non_overlapping():
    events_df, gt_df = generate(seed=2, days=3, interval_minutes=10)
    ts_min, ts_max = events_df["ts"].min(), events_df["ts"].max()
    assert (gt_df["start_ts"] >= ts_min).all()
    assert (gt_df["end_ts"] <= ts_max).all()

    for (_, _), group in gt_df.groupby(["service", "metric_name"]):
        windows = sorted(zip(group["start_ts"], group["end_ts"]))
        for (_, e1), (s2, _) in zip(windows, windows[1:]):
            assert e1 < s2


def test_generate_is_deterministic():
    e1, g1 = generate(seed=7, days=1, interval_minutes=15)
    e2, g2 = generate(seed=7, days=1, interval_minutes=15)
    assert e1["value"].tolist() == e2["value"].tolist()
    assert g1["magnitude"].tolist() == g2["magnitude"].tolist()


def test_values_stay_in_physical_bounds():
    """Epic 1-2 review #6 — uncapped noise produced a negative error_rate."""
    events_df, _ = generate(seed=42, days=14, interval_minutes=5)
    error_rate = events_df.loc[events_df["metric_name"] == "error_rate", "value"]
    cpu = events_df.loc[events_df["metric_name"] == "cpu_pct", "value"]
    assert (error_rate >= 0).all() and (error_rate <= 1).all()
    assert (cpu >= 0).all() and (cpu <= 100).all()


def test_validate_raises_on_a_stubbed_no_op_injection():
    """Epic 1-2 review #4 — the validator must actually catch a no-op, not
    just reject 'differs from zero'. Build a fixture where the recorded
    ground truth claims a real shift but the events never moved."""
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    # Overwrite one window's values back to a flat baseline, leaving the
    # ground-truth row's magnitude as-is — simulates the injection silently
    # not landing while the answer key still claims it did.
    row = gt_df.iloc[0]
    mask = (
        (events_df["service"] == row["service"])
        & (events_df["metric_name"] == row["metric_name"])
        & (events_df["ts"] >= row["start_ts"])
        & (events_df["ts"] <= row["end_ts"])
    )
    baseline = METRICS[row["metric_name"]]["baseline"]
    events_df.loc[mask, "value"] = baseline

    with pytest.raises(AssertionError):
        _validate(events_df, gt_df)


def test_committed_corpus_matches_current_code():
    """The corpus is a committed build artifact — nothing else keeps it in
    sync with the generator after an edit (Epic 1-2 review #9)."""
    from app.db import get_connection, get_ground_truth_connection

    events_df, gt_df = generate(seed=42, days=14, interval_minutes=5)

    conn = get_connection(read_only=True)
    committed_events = conn.execute(
        "SELECT id, ts, service, metric_name, value, level, message FROM events ORDER BY id"
    ).fetchdf()
    conn.close()
    committed_events["ts"] = pd.to_datetime(committed_events["ts"]).astype("datetime64[ns]")
    expected_events = events_df.sort_values("id").reset_index(drop=True)
    expected_events["ts"] = expected_events["ts"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(committed_events.reset_index(drop=True), expected_events)

    gt_conn = get_ground_truth_connection(read_only=True)
    committed_gt = gt_conn.execute(
        "SELECT id, service, metric_name, start_ts, end_ts, anomaly_type, magnitude FROM ground_truth_anomalies ORDER BY id"
    ).fetchdf()
    gt_conn.close()
    committed_gt["start_ts"] = pd.to_datetime(committed_gt["start_ts"]).astype("datetime64[ns]")
    committed_gt["end_ts"] = pd.to_datetime(committed_gt["end_ts"]).astype("datetime64[ns]")
    committed_gt["id"] = committed_gt["id"].astype("int64")
    expected_gt = gt_df.sort_values("id").reset_index(drop=True)
    expected_gt["start_ts"] = expected_gt["start_ts"].astype("datetime64[ns]")
    expected_gt["end_ts"] = expected_gt["end_ts"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(committed_gt.reset_index(drop=True), expected_gt)
