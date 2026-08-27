# Detector Eval Results

Real numbers, run against the committed corpus (`data/ops.duckdb` + `data/ground_truth.duckdb`, seed 42, 14 days / 5-min resolution, 36,288 events, 17 ground-truth anomalies). Reproduce with `python -m eval.eval_detector` — it also writes `eval/detector_results.json`, which this file cites rather than hand-transcribes.

Match definition: any temporal overlap between a detected window and a ground-truth window for the same (service, metric_name) counts as a hit.

## Overall

| Metric | Value |
|---|---|
| Recall | 15/17 = 0.882 |
| Precision | 16/17 = 0.941 |
| F1 | 0.911 |
| False positives | 1 |

16 detections cover 15 distinct ground-truth anomalies — one anomaly (`payments-worker/latency_ms` spike, ground-truth window 9) is covered by two `seasonal_residual` rows separated by a 45-minute sub-threshold gap, wider than `_MERGE_GAP_TOLERANCE`'s 30 minutes, so `merge_detections` correctly leaves them as two rows. (This is not `rolling_zscore`'s onset/recovery self-absorption — `rolling_zscore` doesn't detect this window at all; that fragmentation mode is real and documented but happens not to be what produced this particular case.) Both fragments count as true positives (both genuinely overlap the real anomaly), so this is 16 hits, not 16 anomalies found.

## By anomaly type (recall only)

| Type | Recall |
|---|---|
| spike | 4/4 = 1.000 |
| sustained_drift | 10/10 = 1.000 |
| dip | 1/3 = 0.333 |

**Precision and F1 are not reported per type or per bucket, by construction, not by omission:** a detection carries no ground-truth type label, so attributing a false positive to "the type it should have been" would be meaningless. Stratification is recall-only here; precision is reported once overall and per detector below.

## By anomaly type × magnitude bucket

Bucketed on magnitude **in units of the metric's own noise (σ)**, not raw units — `METRICS` gives each metric a different scale (`latency_ms` std=15.0, `cpu_pct` std=5.0, `error_rate` std=0.1), so a raw-unit split would mostly reflect which metric an anomaly happened to land on, not how subtle it actually was.

| Type / bucket | Recall |
|---|---|
| spike / below_median | 2/2 |
| spike / above_median | 2/2 |
| sustained_drift / below_median | 5/5 |
| sustained_drift / above_median | 5/5 |
| dip / below_median | **0/2** |
| dip / above_median | 1/1 |

**Both missed windows are `error_rate` dips at 1.33σ and 1.96σ — below the detection floor by construction, not a detector failure.** `error_rate`'s tight [0,1] bound forces small dips to avoid flatlining against the floor, a known limitation documented in the architecture doc *before* this eval ran. No z-score-based detector at any reasonable threshold reliably catches a sub-2σ shift. The `dip/above_median` hit (`payments-worker/latency_ms`, an unbounded metric, 4.11σ) shows the detectors work correctly on dips that aren't fighting a floor.

## Per-detector precision (standalone, measured before merging)

| Detector | Precision | Recall (of all 17) |
|---|---|---|
| `rolling_zscore` (window=48, shipped default) | 5/6 = 0.833 | 5/17 |
| `rolling_zscore` (window=144) | **11/11 = 1.000** | **9/17** |
| `seasonal_residual` | 26/26 = 1.000 | 15/17 |

**`rolling_zscore` is corroborating, not additive, on this corpus, at either window.** Every ground-truth window it finds — at window=48 or window=144 — is one `seasonal_residual` already found (0 unique recall at every window size from 24 to 288 samples, measured during Epic 3's review; verified again here that `rolling_zscore`'s window=144 hit set is a strict subset of `seasonal_residual`'s). Its honest contribution is "confirms N of the 15 found windows" — a real, reportable finding about how the two methods relate on this data, not a flaw in either method. `window=144` confirms nearly twice as many (9 vs 5) at perfect standalone precision, which is why the merged table below looks better at that setting too.

## Per merged-row method combination (post-merge — different from the table above)

| Combination | Rows that hit a ground-truth window |
|---|---|
| `seasonal_residual` (standalone rows) | 11/11 |
| `rolling_zscore+seasonal_residual` (both caught it) | 5/5 |
| `rolling_zscore` (standalone rows, post-merge) | 0/1 |

This is **not** `rolling_zscore`'s own precision (that's the table above, 0.833) — after merging, 5 of its 6 pre-merge detections already got folded into a `rolling_zscore+seasonal_residual` row, leaving only its one miss as a standalone post-merge row.

## The window=144 alternative

The architecture doc flagged this as worth reporting: `rolling_zscore` at `window=144` (12h baseline instead of the shipped 4h) measures **strictly better** on this corpus, both standalone (table above) and merged into the full pipeline:

| Config (post-merge, full pipeline) | Recall | Precision | False positives |
|---|---|---|---|
| `window=48` (shipped default) | 15/17 | 16/17 = 0.941 | 1 |
| `window=144` | 15/17 | 16/16 = **1.000** | **0** |

Not switched as the default — the difference is one row on a single 17-anomaly corpus, and both configs were tuned against this same corpus. Reported here rather than silently adopted, per the architecture doc's instruction.

## The one false positive (at the shipped default)

`auth-service/cpu_pct`, 2026-08-08 17:30-17:35, `rolling_zscore` only, score 3.28 — a 2-point noise run at the statistical floor of the `min_run=2` persistence threshold. Not a recovery-phantom artifact: the nearest ground-truth window on that series is six days away. It disappears entirely at `window=144` (see above).

## Known limitations, stated plainly

- `rolling_zscore`'s trailing baseline self-absorbs sustained anomalies longer than its window (48 points / 4 hours) and reports onset/recovery rather than full span — scored here by overlap, never by span coverage.
- The first 4 hours of every series is unscoreable by `rolling_zscore` (rolling-window warm-up: `min_periods=48` on a `shift(1)`'d calc leaves indices 0-47 as NaN). Zero ground-truth windows start in that region on this corpus, so it costs nothing here, but a shorter series or a reseeded corpus with an early anomaly would be affected.
- `error_rate × dip` is a hard detection-floor cell by construction (see above) — reported honestly rather than blended into the overall recall number.
