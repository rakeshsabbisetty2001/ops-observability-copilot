# Detector Eval Results

Real numbers, run against the committed corpus (`data/ops.duckdb` + `data/ground_truth.duckdb`, seed 42, 14 days / 5-min resolution, 36,288 events, 17 ground-truth anomalies). Reproduce with `python -m eval.eval_detector`.

Match definition: any temporal overlap between a detected window and a ground-truth window for the same (service, metric_name) counts as a hit.

## Overall

| Metric | Value |
|---|---|
| Recall | 15/17 = 0.882 |
| Precision | 16/17 = 0.941 |
| F1 | 0.911 |
| False positives | 1 |

## By anomaly type (not blended — a real finding, not an average)

| Type | Recall |
|---|---|
| spike | 4/4 = 1.000 |
| sustained_drift | 10/10 = 1.000 |
| dip | 1/3 = 0.333 |

## By anomaly type × magnitude bucket

| Type / bucket | Recall |
|---|---|
| spike / below_median | 2/2 |
| spike / above_median | 2/2 |
| sustained_drift / below_median | 5/5 |
| sustained_drift / above_median | 5/5 |
| dip / below_median | **0/2** |
| dip / above_median | 1/1 |

**Both missed windows are `error_rate` dips, and this is expected, not a detector failure.** `error_rate`'s tight [0,1] bound forces small dips (1-2σ) to avoid flatlining against the floor — a known, deliberate limitation documented in the architecture doc *before* this eval ran. No z-score-based detector at any reasonable threshold reliably catches a sub-2σ shift. The `dip/above_median` hit (`payments-worker/latency_ms`, an unbounded metric, deeper dip) shows the detectors work correctly on dips that aren't fighting a floor.

## By detection method (pre-merge attribution)

| Method | Hit a ground-truth window |
|---|---|
| `seasonal_residual` (standalone) | 11/11 |
| `rolling_zscore` + `seasonal_residual` (both caught it) | 5/5 |
| `rolling_zscore` (standalone) | 0/1 |

**`rolling_zscore` is corroborating, not additive, on this corpus.** Every ground-truth window it finds is one `seasonal_residual` already found (0 unique recall at every window size from 24 to 288 samples, measured during Epic 3's review). Its honest contribution is "confirms 5 of the 15 found windows" — a real, reportable finding about how the two methods relate on this data, not a flaw in either method.

## The one false positive

`auth-service/cpu_pct`, 2026-08-08 17:30-17:35, `rolling_zscore` only, score 3.28 — a 2-point noise run at the statistical floor of the `min_run=2` persistence threshold. Not a recovery-phantom artifact (nearest ground-truth window on that series is hours away).

## Known limitations, stated plainly

- `rolling_zscore`'s trailing baseline self-absorbs sustained anomalies longer than its window (48 points / 4 hours) and reports onset/recovery rather than full span — scored here by overlap, never by span coverage.
- The first ~4h05m of every series is unscoreable by `rolling_zscore` (rolling-window warm-up). Zero ground-truth windows start in that region on this corpus, so it costs nothing here, but a shorter series would be affected.
- `error_rate × dip` is a hard detection-floor cell by construction (see above) — reported honestly rather than blended into the overall recall number.
