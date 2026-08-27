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


def test_match_no_detections():
    gt = pd.DataFrame([_row("svc", "m", "2026-01-01", "2026-01-01 01:00")])
    det = pd.DataFrame(columns=["service", "metric_name", "start_ts", "end_ts", "method"])
    gt_hit, det_hit = match(det, gt)
    metrics = compute_metrics(gt_hit, det_hit)
    assert metrics["recall"] == 0.0
    assert pd.isna(metrics["precision"])  # n_det == 0, undefined, not silently 0 or 1


def test_match_different_metric_name_does_not_match():
    gt = pd.DataFrame([_row("svc", "latency_ms", "2026-01-01", "2026-01-01 01:00")])
    det = pd.DataFrame([_row("svc", "error_rate", "2026-01-01 00:15", "2026-01-01 00:45", method="a")])
    gt_hit, det_hit = match(det, gt)
    assert not gt_hit.any() and not det_hit.any()


def test_magnitude_buckets_split_within_type_not_across():
    gt = pd.DataFrame(
        [
            {"anomaly_type": "spike", "magnitude": 1.0},
            {"anomaly_type": "spike", "magnitude": 100.0},
            {"anomaly_type": "dip", "magnitude": 500.0},  # would be "above" spike's median if compared globally
        ]
    )
    bucketed = add_magnitude_buckets(gt)
    spike = bucketed[bucketed["anomaly_type"] == "spike"]
    assert set(spike["magnitude_bucket"]) == {"below_median", "above_median"}
    # dip has only 1 sample -> can't split, must not silently inherit a bucket from spike's scale
    assert bucketed.loc[bucketed["anomaly_type"] == "dip", "magnitude_bucket"].iloc[0] == "only"


def test_f1_zero_when_precision_and_recall_both_zero():
    metrics = compute_metrics(pd.Series([False]), pd.Series([False]))
    assert metrics["f1"] == 0.0
