"""Seasonal-residual anomaly detection: estimate each hour-of-day's own
average as a same-time-of-day baseline (from the observed data itself, not
the generator's true parameters — a real detector never sees those), then
flag points whose residual from that baseline deviates more than `threshold`
standard deviations from the residual's own spread, per (service, metric_name).
Catches metrics with daily seasonality that a plain rolling z-score would
misread as normal end-of-cycle drift."""
import pandas as pd

from app.detection._windows import flags_to_windows


def detect(events_df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    rows: list[dict] = []
    for (service, metric), group in events_df.groupby(["service", "metric_name"]):
        g = group.sort_values("ts").reset_index(drop=True)
        hourly_mean = g.groupby(g["ts"].dt.hour)["value"].transform("mean")
        residual = g["value"] - hourly_mean
        std = residual.std()
        z = residual / std if std > 0 else residual * 0.0
        rows.extend(flags_to_windows(g, z.abs() > threshold, z, "seasonal_residual", service, metric))

    return pd.DataFrame(rows, columns=["service", "metric_name", "start_ts", "end_ts", "method", "score", "sample_event_ids"])
