"""Shared helper: turn a per-point flagged/z series into merged detection
windows. Both detectors flag individual points; a real anomaly usually spans
several consecutive points, and the eval matches temporal windows against
ground truth, not individual points."""
import pandas as pd


def flags_to_windows(
    g: pd.DataFrame, flagged: pd.Series, z: pd.Series, method: str, service: str, metric: str
) -> list[dict]:
    """g is sorted by ts with a fresh 0..n-1 index; flagged/z share that index.
    Merges consecutive flagged rows (by position, not by time gap — the
    generator's timestamps are evenly spaced, so consecutive positions are
    consecutive points) into one detection row per contiguous run."""
    idx = g.index[flagged.fillna(False)].tolist()
    if not idx:
        return []

    windows = []
    start = prev = idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
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
