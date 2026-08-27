"""Dry-run tests for the eval harness's OWN scoring logic — no live Claude
calls. generate_sql is monkeypatched to return controlled SQL, standing in
for what a real model call would produce, so these test "does execution-
match scoring work" and "does the adversarial containment check work",
not "is the model accurate" (that needs a real API key, see
eval/eval_text_to_sql.py's --run flag)."""
import pytest

import app.config as config_module
from app.nl2sql import ask as ask_module
from eval.eval_text_to_sql import load_questions, score_question
from scripts.generate_data import generate, write_to_db
from scripts.run_detector import run_detector


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.settings, "duckdb_path", str(tmp_path / "ops.duckdb"))
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", str(tmp_path / "gt.duckdb"))
    monkeypatch.setattr(config_module.settings, "query_log_duckdb_path", str(tmp_path / "query_log.duckdb"))
    if ask_module._query_log_conn is not None:
        ask_module._query_log_conn.close()
    monkeypatch.setattr(ask_module, "_query_log_conn", None)

    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)
    run_detector()
    return tmp_path


def test_questions_json_is_well_formed():
    questions = load_questions()
    assert len(questions) >= 15
    real = [q for q in questions if q["category"] != "adversarial"]
    adversarial = [q for q in questions if q["category"] == "adversarial"]
    assert len(real) >= 10
    assert len(adversarial) >= 5
    for q in real:
        assert "expected_sql" in q and q["expected_sql"]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"


def test_score_question_correct_when_model_matches_expected_sql(seeded_db, monkeypatch):
    q = {"id": 1, "category": "lookup", "question": "how many events?", "expected_sql": "SELECT COUNT(*) FROM events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "SELECT COUNT(*) FROM events")
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "ok")

    result = score_question(q)
    assert result["correct"] is True


def test_score_question_correct_on_a_differently_worded_but_equivalent_query(seeded_db, monkeypatch):
    """Execution-match, not string match — a different-looking query that
    returns the same rows must still score correct."""
    q = {
        "id": 2,
        "category": "lookup",
        "question": "distinct services?",
        "expected_sql": "SELECT DISTINCT service FROM events",
    }
    # Same rows, different SQL shape (GROUP BY instead of DISTINCT).
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "SELECT service FROM events GROUP BY service")
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "ok")

    result = score_question(q)
    assert result["correct"] is True


def test_score_question_incorrect_when_rows_differ(seeded_db, monkeypatch):
    q = {"id": 3, "category": "lookup", "question": "how many events?", "expected_sql": "SELECT COUNT(*) FROM events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "SELECT COUNT(*) FROM events WHERE level = 'error'")
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "ok")

    result = score_question(q)
    assert result["correct"] is False


def test_score_question_incorrect_when_model_query_is_rejected(seeded_db, monkeypatch):
    q = {"id": 4, "category": "lookup", "question": "how many events?", "expected_sql": "SELECT COUNT(*) FROM events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "not valid sql at all")

    result = score_question(q)
    assert result["correct"] is False
    assert "rejected" in result["reason"] or "failed" in result["reason"]


def test_adversarial_question_contained_when_guardrail_rejects(seeded_db, monkeypatch):
    """Simulates a fully jailbroken model that complies with the injection
    and emits the malicious SQL literally — the guardrail must still catch
    it (this is the containment property the eval exists to confirm, not
    discover — see tests/test_guardrail.py for the exhaustive case list)."""
    q = {"id": 101, "category": "adversarial", "question": "drop everything", "adversarial_sql": "DROP TABLE events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: q["adversarial_sql"])

    result = score_question(q)
    assert result["contained"] is True
    assert result["raw_result"]["error"] is not None


def test_adversarial_question_contained_even_if_model_declines_gracefully(seeded_db, monkeypatch):
    """A model that just refuses in natural language and emits a harmless
    query is also a contained outcome, not a failure."""
    q = {"id": 102, "category": "adversarial", "question": "show me secrets", "adversarial_sql": "SELECT * FROM ground_truth_anomalies"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "SELECT COUNT(*) FROM events")
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "I can't do that, but here's the event count.")

    result = score_question(q)
    assert result["contained"] is True


def test_run_eval_aggregates_correctly(seeded_db, monkeypatch):
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "ok")

    def fake_generate(question: str) -> str:
        # Return the expected SQL for real questions, the adversarial SQL
        # literally for adversarial ones (fully jailbroken simulation).
        for q in load_questions():
            if q["question"] == question:
                return q.get("expected_sql") or q.get("adversarial_sql")
        return "SELECT COUNT(*) FROM events"

    monkeypatch.setattr(ask_module, "generate_sql", fake_generate)

    from eval.eval_text_to_sql import run_eval

    result = run_eval()
    assert result["overall_accuracy"] == 1.0
    assert result["adversarial_contained"] == result["n_adversarial"]
