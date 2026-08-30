# AI Ops Observability Copilot

Ask your logs and metrics a question in plain English instead of writing SQL, and catch anomalies before customers do.

**Live demo (UI):** https://ops-observability-copilot.streamlit.app
**Live API:** https://ops-observability-copilot.onrender.com
**API docs:** https://ops-observability-copilot.onrender.com/docs (FastAPI auto-generated)

## The problem

An on-call engineer or SRE needs answers from logs/metrics fast — "which service had the most errors last night," "show me the anomaly that spiked checkout-api's latency" — but that means either knowing the schema and writing SQL under pressure, or waiting on whoever does. And anomaly detection in most stacks is either a fixed threshold that misses real drift or floods you with noise from normal daily/weekly cycles.

This system answers natural-language questions over an events/metrics store by generating and executing SQL (with a guardrail between the model and the database), and separately runs two anomaly detectors (rolling z-score for sudden spikes, seasonal-residual for missed daily-cycle drift) against injected ground-truth anomalies so their precision/recall is a measured number, not a claim.

## Architecture

```
NL question --> Claude (text-to-SQL) --> generated SQL
                                              |
                                              v
                                   guardrail: AST-based table/
                                   statement allowlist (DuckDB's
                                   own json_serialize_sql(), not
                                   regex) + read_only, external-
                                   access-off connection (two
                                   independent layers)
                                              |
                                              v
                                   DuckDB (events, detected_anomalies)
                                              |
                                              v
                                   result --> Claude (plain-language summary)

Metrics stream --> rolling z-score detector    --\
                 --> seasonal-residual detector  --> merged detections --> detected_anomalies
```

- **Backend:** FastAPI. `/ask` for NL questions, `/anomalies` + `/anomalies/{id}` to browse/drill into detected anomalies, `/health` for liveness.
- **Data store:** DuckDB, embedded — no separate managed database service. `data/ops.duckdb` holds the events/metrics corpus and `detected_anomalies`; opened `read_only=True` with `enable_external_access=false` for every query the text-to-SQL path executes, as a physical backstop independent of the app-level guardrail.
- **Text-to-SQL guardrail:** Claude-generated SQL is parsed with DuckDB's own `json_serialize_sql()` into a real AST — not pattern-matched as text — to enforce a table/statement allowlist before anything executes. Two false starts along the way are worth naming: a regex-based table check missed comma joins, and an early CTE-scoping bug let a throwaway `WITH query_log AS (...)` whitelist every real reference to that name elsewhere in the query. Both are why the read-only/no-external-access connection exists as a second, independent layer — never trust the guardrail alone.
- **Anomaly detection:** two independent detectors — rolling z-score (catches sudden spikes) and seasonal-residual (median-per-hour-of-day baseline with MAD-based spread, both outlier-resistant by construction, to catch drift a plain rolling mean/std would misread as normal end-of-cycle noise). Detections are merged across a tolerance window before being written to `detected_anomalies`.
- **UI:** React/TypeScript SPA (`web/`) — a chat interface for NL questions plus a browse/drill-down view over detected anomalies, with real baseline context around each flagged window (not just the flagged points themselves). Replaces an earlier Streamlit UI (`ui/`, being phased out) with the same views plus click-to-drill-down, live filtering, sortable columns, and chart hover tooltips.

## Production concerns handled

- **Defense in depth on the SQL path:** the guardrail's AST-based allowlist is not the only thing stopping a bad query — the DuckDB connection itself is opened read-only with external access disabled, so `COPY TO` / `read_csv_auto` / `ATTACH` and all DML/DDL are blocked at the database level even if the guardrail has a bug. Verified live: 30/30 forged-header requests through a 10/minute rate limit were caught by fixing a real gap, not assumed closed (see below).
- **Rate limiting** keyed on the request's client IP, verified against uvicorn's actual proxy-header handling, not just configured and assumed correct: uvicorn 0.32.1 defaults to trusting `X-Forwarded-For` from any `127.0.0.1` peer, which silently defeated the rate limiter until `--no-proxy-headers` was added to the Dockerfile's `CMD` and re-verified from inside a real container against the real attack.
- **Request body size cap** (`BodySizeLimitMiddleware`), applied consistently to the fixed-SQL `/anomalies` path as well as the NL `/ask` path — an unbounded row `limit` isn't hypothetical here: this project's own detector history produced a 595-row result from a bare 3-sigma threshold before tuning.
- **Structured error handling** that whitelists which fields of a validation error get echoed back to the client (`loc`/`type`/`msg`) rather than blacklisting the ones considered sensitive — closes two classes of leak at once (an unserializable exception object in `ctx`, and a raw input value smuggled into `msg`) instead of only the one that's been observed.
- **Non-root container user.**
- **Structured logging** for every rate-limit rejection, verified end-to-end: an earlier version logged extra fields via `extra={}`, which never renders because no formatter references them by name — confirmed against real `docker logs` output after the fix, not just against the log call.

## Eval results

**Text-to-SQL — live run against the real deployed service and `claude-sonnet-5` (2026-08-29):**
- **100% execution-match accuracy (16/16)** across lookup, aggregation, and anomaly-linked questions — spot-checked against `expected_sql`, not just passing on a lenient comparison mode.
- **9/9 adversarial questions contained.** 2 of 9 got the model to generate SQL referencing something it shouldn't (a `duckdb_tables()` catalog dump, a fully-qualified system-table reference) — both caught by the guardrail. The other 7 were refused in plain language before any SQL was generated.
- Full writeup: [`eval/text_to_sql_results.md`](eval/text_to_sql_results.md).

**Anomaly detection — against a 14-day, 36,288-event synthetic corpus with 17 injected ground-truth anomalies:**
- **Recall 15/17 (0.882), precision 16/17 (0.941), F1 0.911.**
- Perfect recall on spikes (4/4) and sustained drift (10/10); the weak spot is subtle dips below the noise floor (1/3) — reported honestly rather than folded into the headline number.
- Full writeup: [`eval/detector_results.md`](eval/detector_results.md).

Reproduce: `python -m eval.eval_text_to_sql --run` (real API cost, ~$0.50) and `python -m eval.eval_detector`.

## Known limitations (deliberate scope cuts)

- Synthetic corpus, not a real production telemetry feed — the point is a measurable, reproducible ground truth for the eval, not live ingestion.
- No multi-tenant auth — public, IP-rate-limited demo only.
- No durable spend cap in code, same tradeoff as the other portfolio projects — the real backstop is a monthly limit set on the Anthropic API key itself.
- Dip-type anomalies below the noise floor are the detectors' weakest case (1/3 recall) — reported in the eval rather than hidden, not yet addressed.
- No multi-turn conversation memory in `/ask` — each question is answered independently.

## Running locally

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload --no-proxy-headers
```

`--no-proxy-headers` matters even locally — see Production concerns above.

React UI (`web/`, current): `cd web && npm install && npm run dev` (defaults to `http://localhost:8000`; override via `VITE_API_URL`). See `web/README.md` for build/deploy.

Streamlit UI (`ui/`, being phased out): `pip install -r ui/requirements.txt && streamlit run ui/streamlit_app.py` (set `API_URL` to point at your local API).

## Stack

Python 3.12, FastAPI, DuckDB (embedded), Claude Sonnet 5 (text-to-SQL + summarization), pandas/numpy (detectors), React/TypeScript + Vite (UI), Docker + Render (API) + Vercel (UI).
