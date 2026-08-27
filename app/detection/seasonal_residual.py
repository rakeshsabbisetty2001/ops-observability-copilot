"""Seasonal-residual anomaly detection: estimate each hour-of-day's own
average as a same-time-of-day baseline (from the observed data itself, not
the generator's true parameters — a real detector never sees those), then
flag points whose residual from that baseline deviates more than `threshold`
robust standard deviations from the residual's own spread, per
(service, metric_name). Catches metrics with daily seasonality that a plain
rolling z-score would misread as normal end-of-cycle drift.

Uses the MEDIAN (not mean) per hour-of-day and a MAD-based spread (not std) —
both outlier-resistant by construction. A plain mean/std baseline is pulled
by the very anomalies it's supposed to detect against: reproduced directly in
this project's own test suite, where one injected spike shifted its hour's
mean enough to flag an unrelated clean point at the same hour on a different
day with a bit-identical score (Epic 3 review round 1, #3)."""
import pandas as pd

from app.detection._windows import flags_to_windows


def detect(
    events_df: pd.DataFrame, threshold: float = 3.0, min_run: int = 2, extreme_z: float = 8.0
) -> pd.DataFrame:
    rows: list[dict] = []
    for (service, metric), group in events_df.groupby(["service", "metric_name"]):
        g = group.sort_values("ts").reset_index(drop=True)
        hourly_median = g.groupby(g["ts"].dt.hour)["value"].transform("median")
        residual = g["value"] - hourly_median
        # MAD scaled to be std-comparable under normality (1.4826x), same
        # robustness reasoning as the median baseline above. The std fallback
        # below only triggers when MAD is exactly degenerate (a real corner
        # case: a near-constant series where the majority residual value
        # repeats) — that fallback is knowingly NOT robust (it's the same
        # estimator this whole module exists to avoid), so it should only
        # ever engage on a metric that's essentially flat to begin with,
        # where a plain std is a reasonable last resort rather than a
        # meaningful robustness story (Epic 3 review round 2, #8).
        mad = 1.4826 * (residual - residual.median()).abs().median()
        scale = mad if mad > 0 else residual.std()
        z = residual / scale if scale > 0 else residual * 0.0
        rows.extend(
            flags_to_windows(
                g, z.abs() > threshold, z, "seasonal_residual", service, metric, min_run=min_run, extreme_z=extreme_z
            )
        )

    return pd.DataFrame(rows, columns=["service", "metric_name", "start_ts", "end_ts", "method", "score", "sample_event_ids"])
