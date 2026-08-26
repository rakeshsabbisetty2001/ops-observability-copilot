from scripts.generate_data import METRICS, SERVICES, generate


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
