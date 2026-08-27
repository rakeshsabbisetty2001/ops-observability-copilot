import pandas as pd

from eval.eval_detector import add_magnitude_buckets, compute_metrics, match


def _row(service, metric, start, end, **extra):
    return {"service": service, "metric_name": metric, "start_ts": pd.Timestamp(start), "end_ts": pd.Timestamp(end), **extra}


def test_match_known_answer():
    # 3 ground-truth windows: hit by overlap, hit by containment, missed entirely.
    gt = pd.DataFrame(
        [
            _row("svc", "m", "2026-01-01 00:00", "2026-01-01 01:00"),  # overlapping hit
            _row("svc", "m", "2026-01-02 00:00", "2026-01-02 01:00"),  # contained hit
            _row("svc", "m", "2026-01-03 00:00", "2026-01-03 01:00"),  # missed
        ]
    )
    det = pd.DataFrame(
        [
            _row("svc", "m", "2026-01-01 00:30", "2026-01-01 01:30", method="a"),  # overlaps gt row 0
            _row("svc", "m", "2026-01-02 00:15", "2026-01-02 00:45", method="a"),  # contained in gt row 1
            _row("other-svc", "m", "2026-01-03 00:15", "2026-01-03 00:45", method="a"),  # wrong service -> FP
        ]
    )
    gt_hit, det_hit = match(det, gt)
    assert gt_hit.tolist() == [True, True, False]
    assert det_hit.tolist() == [True, True, False]  # wrong-service det never matches

    metrics = compute_metrics(gt_hit, det_hit)
    assert metrics["recall"] == 2 / 3
    assert metrics["precision"] == 2 / 3
    assert metrics["tp_recall"] == 2 and metrics["tp_precision"] == 2


def test_match_marks_every_detection_that_hits_a_multiply_matched_window():
    """Epic 4 review round 1, #2 — an early `break` after the first match
    passed the whole suite while silently dropping the headline precision
    from 0.941 to 0.882 on the real corpus (a real fragmented ground-truth
    window there is matched by two detections). This fixture has a gt window
    matched by TWO detections and asserts both are marked, not just one."""
    gt = pd.DataFrame([_row("svc", "m", "2026-01-01 00:00", "2026-01-01 03:00")])
    det = pd.DataFrame(
        [
            _row("svc", "m", "2026-01-01 00:00", "2026-01-01 01:00", method="a"),  # onset fragment
            _row("svc", "m", "2026-01-01 02:30", "2026-01-01 03:00", method="a"),  # recovery fragment
            _row("svc", "m", "2026-02-01 00:00", "2026-02-01 01:00", method="a"),  # unrelated -> FP
        ]
    )
    gt_hit, det_hit = match(det, gt)
    assert gt_hit.tolist() == [True]
    assert det_hit.tolist() == [True, True, False]

    metrics = compute_metrics(gt_hit, det_hit)
    assert metrics["tp_recall"] == 1  # one gt window covered
    assert metrics["tp_precision"] == 2  # by two separate detections, both real hits


def test_match_marks_every_ground_truth_window_a_wide_detection_spans():
    """Mirror of the case above: one detection spanning two adjacent
    ground-truth windows must recall both, not just the first found."""
    gt = pd.DataFrame(
        [
            _row("svc", "m", "2026-01-01 00:00", "2026-01-01 00:30"),
            _row("svc", "m", "2026-01-01 01:00", "2026-01-01 01:30"),
        ]
    )
    det = pd.DataFrame([_row("svc", "m", "2026-01-01 00:00", "2026-01-01 01:30", method="a")])
    gt_hit, det_hit = match(det, gt)
    assert gt_hit.tolist() == [True, True]
    assert det_hit.tolist() == [True]


def test_match_no_detections():
    gt = pd.DataFrame([_row("svc", "m", "2026-01-01", "2026-01-01 01:00")])
    det = pd.DataFrame(columns=["service", "metric_name", "start_ts", "end_ts", "method"])
    gt_hit, det_hit = match(det, gt)
    metrics = compute_metrics(gt_hit, det_hit)
    assert metrics["recall"] == 0.0
    assert pd.isna(metrics["precision"])  # n_det == 0, undefined, not silently 0 or 1
    # A tempting "simplification" of the f1 guard (`> 0` instead of a bare
    # truthy check) turns this into a fabricated 0.000 instead of an honest
    # NaN — pin it explicitly (Epic 4 review round 1, nit #8).
    assert pd.isna(metrics["f1"])


def test_match_different_metric_name_does_not_match():
    gt = pd.DataFrame([_row("svc", "latency_ms", "2026-01-01", "2026-01-01 01:00")])
    det = pd.DataFrame([_row("svc", "error_rate", "2026-01-01 00:15", "2026-01-01 00:45", method="a")])
    gt_hit, det_hit = match(det, gt)
    assert not gt_hit.any() and not det_hit.any()


def test_magnitude_buckets_split_within_type_not_across():
    gt = pd.DataFrame(
        [
            {"anomaly_type": "spike", "metric_name": "error_rate", "magnitude": 1.0},
            {"anomaly_type": "spike", "metric_name": "error_rate", "magnitude": 100.0},
            # would be "above" spike's median if compared globally
            {"anomaly_type": "dip", "metric_name": "error_rate", "magnitude": 500.0},
        ]
    )
    bucketed = add_magnitude_buckets(gt)
    spike = bucketed[bucketed["anomaly_type"] == "spike"]
    assert set(spike["magnitude_bucket"]) == {"below_median", "above_median"}
    # dip has only 1 sample -> can't split, must not silently inherit a bucket from spike's scale
    assert bucketed.loc[bucketed["anomaly_type"] == "dip", "magnitude_bucket"].iloc[0] == "only"


def test_magnitude_buckets_use_sigma_not_raw_units():
    """Epic 4 review round 1, #3 — bucketing on raw magnitude made the
    'subtlety' split largely a metric_name split, since METRICS gives each
    metric a different noise scale. Two spikes with the SAME raw magnitude
    but on metrics with very different std must land in different buckets
    relative to their own noise, and a larger-in-sigma effect on the
    tighter-noise metric must not be mistaken for the smaller one."""
    gt = pd.DataFrame(
        [
            {"anomaly_type": "spike", "metric_name": "cpu_pct", "magnitude": 10.0},  # cpu_pct std=5.0 -> 2 sigma
            {"anomaly_type": "spike", "metric_name": "latency_ms", "magnitude": 10.0},  # latency_ms std=15.0 -> 0.67 sigma
        ]
    )
    bucketed = add_magnitude_buckets(gt)
    cpu_row = bucketed[bucketed["metric_name"] == "cpu_pct"].iloc[0]
    latency_row = bucketed[bucketed["metric_name"] == "latency_ms"].iloc[0]
    assert cpu_row["magnitude_sigma"] > latency_row["magnitude_sigma"]
    assert cpu_row["magnitude_bucket"] == "above_median"
    assert latency_row["magnitude_bucket"] == "below_median"


def test_f1_zero_when_precision_and_recall_both_zero():
    metrics = compute_metrics(pd.Series([False]), pd.Series([False]))
    assert metrics["f1"] == 0.0
