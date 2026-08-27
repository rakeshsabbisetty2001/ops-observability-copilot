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
from pathlib import Path

import pandas as pd

from app.db import get_connection, get_ground_truth_connection
from app.detection import rolling_zscore, seasonal_residual
from scripts.generate_data import METRICS
from scripts.run_detector import merge_detections

_RESULTS_JSON_PATH = Path(__file__).with_name("detector_results.json")


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
    window 9, payments-worker/latency_ms spike, is covered by two
    seasonal_residual rows separated by a 45-minute sub-threshold gap wider
    than _MERGE_GAP_TOLERANCE's 30 minutes, so merge_detections correctly
    leaves them as two rows — not rolling_zscore self-absorption, which
    doesn't touch this window at all), and a detection spanning several
    ground-truth windows must mark all of those too (doesn't occur on this
    corpus, verified separately)."""
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
    # Reused across the window=144 sweep below (round 2, nit #8: was
    # computed twice) and computed from the SAME detect() calls used to
    # build the window=144 merge, so the two sections can't drift apart.
    seasonal_df = seasonal_residual.detect(events_df)
    zscore_df = rolling_zscore.detect(events_df)
    zscore_144_df = rolling_zscore.detect(events_df, window=144)

    by_method_premerge = {}
    for name, pre_merge_df in (
        ("rolling_zscore (window=48, shipped default)", zscore_df),
        ("rolling_zscore (window=144)", zscore_144_df),
        ("seasonal_residual", seasonal_df),
    ):
        pre_gt_hit, pre_det_hit = match(pre_merge_df, gt)
        # compute_metrics, not a hand-rolled ratio (round 2, nit #9) — reuses
        # the same NaN-on-undefined convention nit #8 pinned with a test.
        by_method_premerge[name] = compute_metrics(pre_gt_hit, pre_det_hit)

    # Tripwire: this eval's own detect() calls must actually be the same
    # detector run that produced the committed detected_anomalies, or the
    # per-detector breakdown above silently describes a different detector
    # than the overall numbers below it (round 2, #3). Compares CONTENT, not
    # just row count — a bare length check has real blind spots on this
    # corpus (round 3, #1: rolling_zscore window in {60, 72, 96, 192} all
    # also produce exactly 17 merged rows, with different content, and the
    # length-only version of this assert did not fire against any of them).
    def _row_key(df: pd.DataFrame) -> list:
        cols = df[["service", "metric_name", "start_ts", "end_ts", "method"]].copy()
        cols["start_ts"] = pd.to_datetime(cols["start_ts"])
        cols["end_ts"] = pd.to_datetime(cols["end_ts"])
        return sorted(cols.itertuples(index=False, name=None))

    reconstructed = merge_detections(zscore_df, seasonal_df)
    if _row_key(reconstructed) != _row_key(det):
        raise RuntimeError(
            f"eval_detector's own detect() calls produced {len(reconstructed)} merged rows "
            f"with different content than detected_anomalies' {len(det)} rows — "
            f"scripts/run_detector.py's parameters have drifted from this eval's; re-run "
            f"python -m scripts.run_detector or update this module"
        )

    by_method_postmerge = {}
    for method, sub in det.groupby("method"):
        sub_hit = det_hit.loc[sub.index]
        by_method_postmerge[method] = {"n": len(sub), "hits": int(sub_hit.sum())}

    # Architecture doc's explicit Epic 4 ask: report the window=144 sweep
    # point, since it measures strictly better on this corpus but wasn't
    # made the default (Epic 4 review round 1, #4). This is the POST-MERGE
    # figure — the standalone rolling_zscore@144 number is in
    # by_method_premerge above (round 2, #2: these two numbers are
    # different quantities and both are worth reporting).
    merged_144 = merge_detections(zscore_144_df, seasonal_df)
    gt_hit_144, det_hit_144 = match(merged_144, gt)
    window_144_merged = compute_metrics(gt_hit_144, det_hit_144)

    return {
        "overall": compute_metrics(gt_hit, det_hit),
        "by_type": by_type,
        "by_type_bucket": by_type_bucket,
        "by_method_premerge": by_method_premerge,
        "by_method_postmerge": by_method_postmerge,
        "window_144_merged": window_144_merged,
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

    print("\nPer-detector precision, standalone (measured BEFORE merge_detections' '+' join):")
    for name, m in result["by_method_premerge"].items():
        print(f"  {name}: {m['tp_precision']}/{m['n_det']} precision = {m['precision']:.3f}, "
              f"{m['tp_recall']}/{m['n_gt']} recall = {m['recall']:.3f}")

    print("\nPer merged-row method combination (POST-merge; a '+' row is one both detectors caught):")
    for method, m in result["by_method_postmerge"].items():
        print(f"  {method}: {m['hits']}/{m['n']} rows hit a ground-truth window")

    w144 = result["window_144_merged"]
    print(f"\nMerged table at rolling_zscore window=144 (not the shipped default), post-merge: "
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
    # NaN precision/recall (only reachable on an empty corpus, not this one)
    # would serialize as a bare `NaN` token, which json.dump accepts but
    # which isn't valid per the JSON spec — a known, currently-unreachable
    # edge case (round 2, nit #7), not fixed since nothing on this corpus
    # exercises it and a real fix (None-coercion) would obscure the NaN
    # convention compute_metrics deliberately uses.
    json_result = {
        "overall": result["overall"],
        "by_type": result["by_type"],
        "by_type_bucket": {f"{t}|{b}": m for (t, b), m in result["by_type_bucket"].items()},
        "by_method_premerge": result["by_method_premerge"],
        "by_method_postmerge": result["by_method_postmerge"],
        "window_144_merged": result["window_144_merged"],
        "false_positives": result["false_positives"][["service", "metric_name", "start_ts", "end_ts", "method", "score"]].to_dict("records"),
        "missed": result["missed"][["service", "metric_name", "anomaly_type", "magnitude"]].to_dict("records"),
    }
    with open(_RESULTS_JSON_PATH, "w") as f:
        json.dump(json_result, f, indent=2, default=str)
    print(f"\nWrote {_RESULTS_JSON_PATH}")
