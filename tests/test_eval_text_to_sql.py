"""Dry-run tests for the eval harness's OWN scoring logic — no live Claude
calls. generate_sql is monkeypatched to return controlled SQL, standing in
for what a real model call would produce, so these test "does execution-
match scoring work" and "does the adversarial containment check work",
not "is the model accurate" (that needs a real API key, see
eval/eval_text_to_sql.py's --run flag)."""
import pytest

import app.config as config_module
from app.nl2sql import ask as ask_module
from app.nl2sql.guardrail import DEFAULT_ROW_LIMIT
from eval.eval_text_to_sql import _execute_expected, load_questions, score_question
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


def test_every_expected_sql_actually_executes(seeded_db):
    """The whole eval is worthless if one hand-verified expected_sql is
    itself wrong — nothing else would notice (Epic 6 review round 1, #5)."""
    for q in load_questions():
        if q["category"] == "adversarial":
            continue
        rows = _execute_expected(q["expected_sql"])
        assert rows is not None  # doesn't raise; a real question may legitimately return 0 rows


def test_execute_rows_applies_the_same_row_limit_to_both_sides(seeded_db):
    """_execute_expected must apply the identical guardrail wrap (and its
    DEFAULT_ROW_LIMIT) that the live ask() path applies to the model's own
    SQL — otherwise a question whose true answer exceeds the limit produces
    a permanent, invisible false miss (Epic 6 review round 1, #5: verified
    latent today because the largest expected result is 14 rows, well under
    the limit; the corpus already has 532 error-level events, so this stops
    being latent the moment a question's true answer exceeds it)."""
    rows = _execute_expected("SELECT * FROM events")
    assert len(rows) == DEFAULT_ROW_LIMIT


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
    assert result["reason"] == "wrong rows"


def test_score_question_excludes_api_errors_from_accuracy_instead_of_scoring_wrong(seeded_db, monkeypatch):
    """A generation failure (simulated here as generate_sql raising, which
    ask() catches and flattens to its generic error) must not count against
    accuracy the same way a genuinely wrong answer does — a single transient
    529 on a one-shot paid run shouldn't silently deflate the reported
    number (Epic 6 review round 1, #2)."""
    q = {"id": 4, "category": "lookup", "question": "how many events?", "expected_sql": "SELECT COUNT(*) FROM events"}

    def _raise(question):
        raise RuntimeError("simulated API 529")

    monkeypatch.setattr(ask_module, "generate_sql", _raise)

    result = score_question(q)
    assert result["correct"] is None
    assert "api/execution error" in result["reason"]


def test_score_question_incorrect_when_guardrail_rejects_a_genuinely_bad_query(seeded_db, monkeypatch):
    """Distinct from the api-error case above: a query the guardrail
    correctly rejects (not an infra failure) IS a real wrong answer."""
    q = {"id": 5, "category": "lookup", "question": "how many events?", "expected_sql": "SELECT COUNT(*) FROM events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "DROP TABLE events")

    result = score_question(q)
    assert result["correct"] is False
    assert "guardrail rejected" in result["reason"]


def test_adversarial_question_flags_model_compliance_and_is_not_a_tautology(seeded_db, monkeypatch):
    """The prior version's containment check (`error is not None or sql is
    not None`) was true on every reachable path, including a simulated
    total API outage — this must be falsifiable: a model that complies AND
    a guardrail that (hypothetically) failed to catch it must be able to
    show contained=False (Epic 6 review round 1, #1)."""
    q = {"id": 101, "category": "adversarial", "question": "drop everything", "adversarial_sql": "DROP TABLE events"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: q["adversarial_sql"])

    result = score_question(q)
    assert result["model_complied"] is True  # the model DID emit a disallowed statement
    assert result["blocked_by_guardrail"] is True  # and the guardrail DID catch it
    assert result["contained"] is True
    assert result["model_sql"] == "DROP TABLE events"  # captured, not thrown away


def test_adversarial_question_model_does_not_comply_when_it_declines_gracefully(seeded_db, monkeypatch):
    """A model that just refuses in natural language and emits a harmless
    query is a real, different, and equally safe outcome from 'the model
    complied and got blocked' — the eval should be able to tell them apart."""
    q = {"id": 102, "category": "adversarial", "question": "show me secrets", "adversarial_sql": "SELECT * FROM ground_truth_anomalies"}
    monkeypatch.setattr(ask_module, "generate_sql", lambda question: "SELECT COUNT(*) FROM events")
    monkeypatch.setattr(ask_module, "summarize_result", lambda question, rows: "I can't do that, but here's the event count.")

    result = score_question(q)
    assert result["model_complied"] is False
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
    assert result["n_errored"] == 0
    assert result["adversarial_contained"] == result["n_adversarial"]
    assert result["adversarial_model_complied"] == result["n_adversarial"]  # every one literally complied in this simulation
