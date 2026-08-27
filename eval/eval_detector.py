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
import json

import pandas as pd

from app.db import get_connection, get_ground_truth_connection
from app.detection import rolling_zscore, seasonal_residual
from scripts.generate_data import METRICS
from scripts.run_detector import merge_detections


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = get_connection(read_only=True)
    det = conn.execute(
        "SELECT service, metric_name, start_ts, end_ts, method, score FROM detected_anomalies"
    ).fetchdf()
    events_df = conn.execute("SELECT * FROM events ORDER BY ts").fetchdf()
    conn.close()

    gt_conn = get_ground_truth_connection(read_only=True)
    gt = gt_conn.execute(
        "SELECT id, service, metric_name, start_ts, end_ts, anomaly_type, magnitude FROM ground_truth_anomalies"
    ).fetchdf()
    gt_conn.close()

    for df in (det, gt, events_df):
        for col in ("start_ts", "end_ts", "ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return det, gt, events_df


def match(det: pd.DataFrame, gt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (gt_hit, det_hit): per-row bool Series, aligned to det/gt's own
    index. No early exit on either loop — a ground-truth window matched by
    several detections must mark all of them (real case on this corpus: gt
    window 9 is split across two rolling_zscore onset/recovery fragments),
    and a detection spanning several ground-truth windows must mark all of
    those too (doesn't occur on this corpus, verified separately)."""
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
    # NaN's truthiness makes `if (precision + recall)` correctly fall through
    # to NaN here rather than fabricate 0.0 when either is undefined — do not
    # "simplify" this to `> 0`, which silently turns an undefined F1 into a
    # real-looking zero (Epic 4 review round 1, nit #8).
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
    """Below/above-median split WITHIN each anomaly type, on magnitude
    EXPRESSED IN SIGMA OF THE METRIC'S OWN NOISE (magnitude / METRICS[metric]
    ["std"]), not raw units. Raw-unit magnitude isn't comparable even within
    one anomaly_type, because METRICS gives each metric its own scale
    (latency_ms std=15.0, cpu_pct std=5.0, error_rate std=0.1) — bucketing on
    raw units made this largely a metric_name bucket wearing a subtlety
    label (Epic 4 review round 1, #3: 4 of 17 windows moved bucket once
    re-measured in sigma, and the shipped 'spike' split was exact metric
    identity rather than a real subtle/obvious split)."""
    gt = gt.copy()
    gt["magnitude_sigma"] = gt.apply(lambda r: r["magnitude"] / METRICS[r["metric_name"]]["std"], axis=1)
    gt["magnitude_bucket"] = "only"  # types with <2 samples can't split
    for _, sub in gt.groupby("anomaly_type"):
        if len(sub) < 2:
            continue
        median = sub["magnitude_sigma"].median()
        gt.loc[sub.index, "magnitude_bucket"] = sub["magnitude_sigma"].apply(
            lambda m, med=median: "below_median" if m <= med else "above_median"
        )
    return gt


def run_eval() -> dict:
    det, gt, events_df = load_data()
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

    # Per-detector precision, run BEFORE merge_detections joins method names
    # with "+" — grouping the shipped (post-merge) `method` column is a
    # different, and much easier to misread, quantity (Epic 4 review round
    # 1, #1: the post-merge view reports rolling_zscore as 0/1, implying
    # 0.000 precision, when its real standalone precision is 5/6 = 0.833).
    by_method_premerge = {}
    for name, pre_merge_df in (
        ("rolling_zscore", rolling_zscore.detect(events_df)),
        ("seasonal_residual", seasonal_residual.detect(events_df)),
    ):
        _, pre_hit = match(pre_merge_df, gt)
        by_method_premerge[name] = {
            "n": len(pre_merge_df),
            "tp": int(pre_hit.sum()),
            "precision": (pre_hit.sum() / len(pre_merge_df)) if len(pre_merge_df) else float("nan"),
        }

    by_method_postmerge = {}
    for method, sub in det.groupby("method"):
        sub_hit = det_hit.loc[sub.index]
        by_method_postmerge[method] = {"n": len(sub), "hits": int(sub_hit.sum())}

    # Architecture doc's explicit Epic 4 ask: report the window=144 sweep
    # point, since it measures strictly better on this corpus but wasn't
    # made the default (Epic 4 review round 1, #4).
    zscore_144 = rolling_zscore.detect(events_df, window=144)
    merged_144 = merge_detections(zscore_144, seasonal_residual.detect(events_df))
    gt_hit_144, det_hit_144 = match(merged_144, gt)
    window_144 = compute_metrics(gt_hit_144, det_hit_144)

    return {
        "overall": compute_metrics(gt_hit, det_hit),
        "by_type": by_type,
        "by_type_bucket": by_type_bucket,
        "by_method_premerge": by_method_premerge,
        "by_method_postmerge": by_method_postmerge,
        "window_144": window_144,
        "false_positives": det.loc[~det_hit],
        "missed": gt.loc[~gt_hit],
    }


if __name__ == "__main__":
    result = run_eval()
    o = result["overall"]
    print(f"Overall: recall {o['tp_recall']}/{o['n_gt']} = {o['recall']:.3f}, "
          f"precision {o['tp_precision']}/{o['n_det']} = {o['precision']:.3f}, f1={o['f1']:.3f}")

    print("\nBy anomaly type (recall only — see note on precision below):")
    for t, m in result["by_type"].items():
        print(f"  {t}: {m['hits']}/{m['n']} = {m['recall']:.3f}")

    print("\nBy anomaly type x magnitude bucket (bucketed in sigma of the metric's own noise):")
    for (t, b), m in result["by_type_bucket"].items():
        print(f"  {t}/{b}: {m['hits']}/{m['n']} = {m['recall']:.3f}")

    print("\nPer-detector precision (measured BEFORE merge_detections' '+' join):")
    for name, m in result["by_method_premerge"].items():
        print(f"  {name}: {m['tp']}/{m['n']} = {m['precision']:.3f}")

    print("\nPer merged-row method combination (POST-merge; a '+' row is one both detectors caught):")
    for method, m in result["by_method_postmerge"].items():
        print(f"  {method}: {m['hits']}/{m['n']} rows hit a ground-truth window")

    w144 = result["window_144"]
    print(f"\nrolling_zscore window=144 sweep point (not the shipped default): "
          f"recall {w144['tp_recall']}/{w144['n_gt']} = {w144['recall']:.3f}, "
          f"precision {w144['tp_precision']}/{w144['n_det']} = {w144['precision']:.3f}")

    print(f"\nFalse positives ({len(result['false_positives'])}):")
    if not result["false_positives"].empty:
        print(result["false_positives"][["service", "metric_name", "start_ts", "end_ts", "method", "score"]].to_string(index=False))

    print(f"\nMissed ground-truth windows ({len(result['missed'])}):")
    if not result["missed"].empty:
        print(result["missed"][["service", "metric_name", "anomaly_type", "magnitude"]].to_string(index=False))

    # Written for reproducibility (Epic 4 review round 1, #6) — the .md is
    # prose that cites this, not a hand-transcription nothing else checks.
    json_result = {
        "overall": result["overall"],
        "by_type": result["by_type"],
        "by_type_bucket": {f"{t}|{b}": m for (t, b), m in result["by_type_bucket"].items()},
        "by_method_premerge": result["by_method_premerge"],
        "by_method_postmerge": result["by_method_postmerge"],
        "window_144": result["window_144"],
        "false_positives": result["false_positives"][["service", "metric_name", "start_ts", "end_ts", "method", "score"]].astype(str).to_dict("records"),
        "missed": result["missed"][["service", "metric_name", "anomaly_type", "magnitude"]].to_dict("records"),
    }
    with open("eval/detector_results.json", "w") as f:
        json.dump(json_result, f, indent=2, default=str)
    print("\nWrote eval/detector_results.json")
