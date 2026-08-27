"""Detector eval: precision/recall/F1 against ground truth, stratified by
anomaly type and magnitude bucket (not blended into one number — a detector
that's great at spikes and blind to subtle drifts should look like that, not
average out to "decent").

Match definition: any temporal overlap between a detected window and a
ground-truth window for the same (service, metric_name) counts as a hit —
the same definition used throughout Epic 3's review process, so these
numbers are consistent with everything already measured and documented
there (recall 15/17, precision 16/17 on the current corpus).

Usage: python -m eval.eval_detector
"""
import pandas as pd

from app.db import get_connection, get_ground_truth_connection


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = get_connection(read_only=True)
    det = conn.execute(
        "SELECT service, metric_name, start_ts, end_ts, method, score FROM detected_anomalies"
    ).fetchdf()
    conn.close()

    gt_conn = get_ground_truth_connection(read_only=True)
    gt = gt_conn.execute(
        "SELECT id, service, metric_name, start_ts, end_ts, anomaly_type, magnitude FROM ground_truth_anomalies"
    ).fetchdf()
    gt_conn.close()

    for df in (det, gt):
        df["start_ts"] = pd.to_datetime(df["start_ts"])
        df["end_ts"] = pd.to_datetime(df["end_ts"])
    return det, gt


def match(det: pd.DataFrame, gt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (gt_hit, det_hit): per-row bool Series, aligned to det/gt's own index."""
    gt_hit = pd.Series(False, index=gt.index)
    det_hit = pd.Series(False, index=det.index)
    for gi, g in gt.iterrows():
        for di, d in det.iterrows():
            if (
                d["service"] == g["service"]
                and d["metric_name"] == g["metric_name"]
                and _overlaps(d["start_ts"], d["end_ts"], g["start_ts"], g["end_ts"])
            ):
                gt_hit[gi] = True
                det_hit[di] = True
    return gt_hit, det_hit


def compute_metrics(gt_hit: pd.Series, det_hit: pd.Series) -> dict:
    n_gt, n_det = len(gt_hit), len(det_hit)
    tp_recall, tp_precision = int(gt_hit.sum()), int(det_hit.sum())
    recall = tp_recall / n_gt if n_gt else float("nan")
    precision = tp_precision / n_det if n_det else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_gt": n_gt,
        "n_det": n_det,
        "tp_recall": tp_recall,
        "tp_precision": tp_precision,
    }


def add_magnitude_buckets(gt: pd.DataFrame) -> pd.DataFrame:
    """Below/above-median split WITHIN each anomaly type — comparing a dip's
    magnitude to a drift's wouldn't be a fair "subtle vs obvious" split,
    since the two types' magnitude are drawn from different scales."""
    gt = gt.copy()
    gt["magnitude_bucket"] = "only"  # types with <2 samples can't split
    for _, sub in gt.groupby("anomaly_type"):
        if len(sub) < 2:
            continue
        median = sub["magnitude"].median()
        gt.loc[sub.index, "magnitude_bucket"] = sub["magnitude"].apply(
            lambda m, med=median: "below_median" if m <= med else "above_median"
        )
    return gt


def run_eval() -> dict:
    det, gt = load_data()
    gt_hit, det_hit = match(det, gt)
    gt = add_magnitude_buckets(gt)

    by_type = {}
    for anomaly_type, sub in gt.groupby("anomaly_type"):
        sub_hit = gt_hit.loc[sub.index]
        by_type[anomaly_type] = {"n": len(sub), "hits": int(sub_hit.sum()), "recall": sub_hit.mean()}

    by_type_bucket = {}
    for (anomaly_type, bucket), sub in gt.groupby(["anomaly_type", "magnitude_bucket"]):
        sub_hit = gt_hit.loc[sub.index]
        by_type_bucket[(anomaly_type, bucket)] = {
            "n": len(sub),
            "hits": int(sub_hit.sum()),
            "recall": sub_hit.mean(),
        }

    by_method = {}
    for method, sub in det.groupby("method"):
        sub_hit = det_hit.loc[sub.index]
        by_method[method] = {"n": len(sub), "hits": int(sub_hit.sum())}

    return {
        "overall": compute_metrics(gt_hit, det_hit),
        "by_type": by_type,
        "by_type_bucket": by_type_bucket,
        "by_method": by_method,
        "false_positives": det.loc[~det_hit],
        "missed": gt.loc[~gt_hit],
    }


if __name__ == "__main__":
    result = run_eval()
    o = result["overall"]
    print(f"Overall: recall {o['tp_recall']}/{o['n_gt']} = {o['recall']:.3f}, "
          f"precision {o['tp_precision']}/{o['n_det']} = {o['precision']:.3f}, f1={o['f1']:.3f}")

    print("\nBy anomaly type:")
    for t, m in result["by_type"].items():
        print(f"  {t}: {m['hits']}/{m['n']} = {m['recall']:.3f}")

    print("\nBy anomaly type x magnitude bucket:")
    for (t, b), m in result["by_type_bucket"].items():
        print(f"  {t}/{b}: {m['hits']}/{m['n']} = {m['recall']:.3f}")

    print("\nBy detection method (own precision, pre-merge attribution):")
    for method, m in result["by_method"].items():
        print(f"  {method}: {m['hits']}/{m['n']} hit a ground-truth window")

    print(f"\nFalse positives ({len(result['false_positives'])}):")
    if not result["false_positives"].empty:
        print(result["false_positives"][["service", "metric_name", "start_ts", "end_ts", "method"]].to_string(index=False))

    print(f"\nMissed ground-truth windows ({len(result['missed'])}):")
    if not result["missed"].empty:
        print(result["missed"][["service", "metric_name", "anomaly_type", "magnitude"]].to_string(index=False))
