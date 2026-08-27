# AI Ops Observability Copilot

NL query + anomaly detection over synthetic logs/metrics. Project 4 of a 4-project AI Engineer portfolio.

## Setup
1. `python -m venv .venv` then activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
4. `uvicorn app.main:app --reload`

## Stack
Python 3.12, FastAPI, DuckDB (embedded analytical DB), Claude (text-to-SQL), Streamlit UI.

Status: Epics 1-3 done (foundation, synthetic data + ground truth, anomaly detectors). See `secondBrain` vault for the full architecture/feature plan and review history.
