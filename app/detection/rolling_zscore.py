"""Rolling z-score anomaly detection: flag points whose value deviates more
than `threshold` standard deviations from a trailing rolling window's own
mean/std, per (service, metric_name).

Known limitation, not a bug: a trailing baseline only ever excludes the point
under test, not that point's predecessors (see the shift(1) comment below).
Once a sustained anomaly has run for `window` points, the whole baseline is
itself anomalous and z collapses — coverage is pinned at roughly `window`'s
warm-up worth of points regardless of how long the real anomaly lasts, and
the return to normal afterward can itself score as a (phantom) anomaly since
the baseline hasn't caught up yet. Reports onset and recovery, not the full
span. Epic 4 must score by temporal overlap with ground truth, never by
span-coverage percentage (Epic 3 review round 1, #2)."""
import numpy as np
import pandas as pd

from app.detection._windows import flags_to_windows


def detect(
    events_df: pd.DataFrame, window: int = 48, threshold: float = 3.0, min_run: int = 2
) -> pd.DataFrame:
    # window=48 (4 hours at 5-min resolution), not 12 (1 hour): a 1-hour
    # trailing baseline is comparable in length to this corpus's anomaly
    # windows (up to 36 points), so the baseline "chases" the anomaly and
    # rarely clears threshold at all once min_run=2 is applied — measured on
    # the real corpus, window=12 contributed 7 false positives and 0 unique
    # recall (its one true positive was already found by seasonal_residual);
    # window=48 recovers 5 of 17 windows on its own at 0.833 precision (Epic
    # 3 review round 2, #1).
    rows: list[dict] = []
    for (service, metric), group in events_df.groupby(["service", "metric_name"]):
        g = group.sort_values("ts").reset_index(drop=True)
        # shift(1) before rolling: the baseline must exclude the point being
        # tested, or a genuine outlier inflates the very std used to judge it
        # (measured on the real corpus: a real injected spike scored z=2.78,
        # just under a 3.0 threshold, purely from including itself).
        prior = g["value"].shift(1)
        roll_mean = prior.rolling(window=window, min_periods=window).mean()
        roll_std = prior.rolling(window=window, min_periods=window).std()
        z = (g["value"] - roll_mean) / roll_std.replace(0, np.nan)
        rows.extend(
            flags_to_windows(g, z.abs() > threshold, z, "rolling_zscore", service, metric, min_run=min_run)
        )

    return pd.DataFrame(rows, columns=["service", "metric_name", "start_ts", "end_ts", "method", "score", "sample_event_ids"])
