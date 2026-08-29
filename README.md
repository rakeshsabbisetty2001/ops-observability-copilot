# AI Ops Observability Copilot

NL query + anomaly detection over synthetic logs/metrics. Project 4 of a 4-project AI Engineer portfolio.

## Setup
1. `python -m venv .venv` then activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
4. `uvicorn app.main:app --reload --no-proxy-headers`
   (`--no-proxy-headers` matters even locally: uvicorn's default `proxy_headers=True`
   trusts `X-Forwarded-For` from a `127.0.0.1` peer, which defeats `TRUST_PROXY=false`'s
   rate-limit protection — verified live, 30/30 forged-header requests got through a
   10/minute limit without this flag. See the Dockerfile's `CMD` comment. Epic 8 review round 2, Medium #1.)

## Stack
Python 3.12, FastAPI, DuckDB (embedded analytical DB), Claude (text-to-SQL), Streamlit UI.

Status: all 8 epics done and deployed. Live API: https://ops-observability-copilot.onrender.com. Live UI: https://ops-observability-copilot.streamlit.app. See [`eval/detector_results.md`](eval/detector_results.md) for detector precision/recall (15/17 recall, 16/17 precision) and [`eval/text_to_sql_results.md`](eval/text_to_sql_results.md) for the text-to-SQL eval (100% execution-match accuracy, 9/9 adversarial questions contained), and the `secondBrain` vault for the full architecture/feature plan and review history.
