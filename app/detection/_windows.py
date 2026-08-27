"""Shared helper: turn a per-point flagged/z series into merged detection
windows. Both detectors flag individual points; a real anomaly usually spans
several consecutive points, and the eval matches temporal windows against
ground truth, not individual points."""
import pandas as pd


def flags_to_windows(
    g: pd.DataFrame,
    flagged: pd.Series,
    z: pd.Series,
    method: str,
    service: str,
    metric: str,
    min_run: int = 2,
    extreme_z: float = 8.0,
) -> list[dict]:
    """g is sorted by ts with a fresh 0..n-1 index; flagged/z share that index.
    Merges consecutive flagged rows (by position, not by time gap — the
    generator's timestamps are evenly spaced, so consecutive positions are
    consecutive points) into one detection row per contiguous run of at least
    `min_run` points — OR a single point whose |z| clears `extreme_z` on its
    own, so a single-sample outage/spike isn't unconditionally invisible.

    min_run defaults to 2, not 1: at a plain 3.0 threshold, a single flagged
    point is indistinguishable from noise (measured false-positive rate on
    pure Gaussian noise is a Student-t tail, not normal — ~1.4% per point,
    not the ~0.27% a naive normal-theory reading suggests) but a real
    anomaly spans several points, so requiring at least 2 consecutive flags
    buys roughly 10x precision for a small recall cost (Epic 3 review round
    1, #1 — measured end to end on the real corpus: 595->34 detections,
    precision 0.074->0.794, recall 15/17->13/17).

    The extreme_z escape exists because min_run=2 otherwise makes an
    arbitrarily large single-point spike undetectable by construction — not
    a small recall cost, an unconditional blind spot (Epic 3 review round 2,
    #4, measured: a 40-sigma single-point spike produced zero detections
    from either detector). Measured cost of the escape on 80,000 points of
    clean noise at threshold=3.0: |z|>=8 fires ~1 point in 80,000 (~0.0013%).
    """
    idx = g.index[flagged].tolist()
    if not idx:
        return []

    windows = []
    start = prev = idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        if prev - start + 1 >= min_run or float(z.loc[start:prev].abs().max()) >= extreme_z:
            seg = g.loc[start:prev]
            windows.append(
                {
                    "service": service,
                    "metric_name": metric,
                    "start_ts": seg["ts"].iloc[0],
                    "end_ts": seg["ts"].iloc[-1],
                    "method": method,
                    "score": float(z.loc[start:prev].abs().max()),
                    "sample_event_ids": seg["id"].tolist()[:5],
                }
            )
        if i is not None:
            start = prev = i
    return windows
