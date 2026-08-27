from fastapi.testclient import TestClient

import app.config as config_module
from app.main import app
from app.nl2sql import ask as ask_module
from scripts.generate_data import generate, write_to_db
from scripts.run_detector import run_detector

client = TestClient(app)


def test_ask_endpoint_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.settings, "duckdb_path", str(tmp_path / "ops.duckdb"))
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", str(tmp_path / "gt.duckdb"))
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)
    run_detector()

    monkeypatch.setattr(ask_module, "generate_sql", lambda q: "SELECT * FROM events LIMIT 3")
    monkeypatch.setattr(ask_module, "summarize_result", lambda q, rows: "3 rows found.")

    response = client.post("/ask", json={"question": "show me events"})
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 3
    assert body["answer"] == "3 rows found."
    assert body["error"] is None


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # pydantic min_length


def test_ask_endpoint_rejects_oversized_question():
    response = client.post("/ask", json={"question": "x" * 2000})
    assert response.status_code == 422  # pydantic max_length
