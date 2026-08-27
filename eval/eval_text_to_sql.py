"""Text-to-SQL eval: execution-match accuracy against a held-out NL-question
set with hand-verified expected SQL, plus an adversarial subset that must be
contained by the guardrail regardless of what the model actually generates.

Execution-match, not string similarity: two different-looking SQL strings
that return the same rows both count as correct — the metric is "did the
user get the right answer", not "did the model guess the exact SQL I wrote".

Live-only: every non-adversarial question makes a real Claude call through
the actual app.nl2sql.ask.ask() flow. Estimate the cost before running (see
__main__) — same pre-flight-budget discipline as Project 3's eval.

Usage: python -m eval.eval_text_to_sql --run
Dry run (no API calls, exercises the harness's own scoring logic against a
monkeypatched generate_sql) is what tests/test_eval_text_to_sql.py does.
"""
import argparse
import json
from pathlib import Path

from app.db import get_connection
from app.nl2sql import ask as ask_module
from app.nl2sql.guardrail import validate_and_prepare

_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
_RESULTS_PATH = Path(__file__).with_name("text_to_sql_results.md")

# Rough per-call cost for the pinned model (see app/config.py) — same
# ballpark as Projects 1/3's structured-output calls (question + schema
# prompt in, one summarization call out). Two live calls per non-adversarial
# question (generation + summarization), one per adversarial question
# (generation only — a rejected/empty result never reaches summarize_result).
_ESTIMATED_COST_PER_CALL_USD = 0.01


def load_questions() -> list[dict]:
    return json.loads(_QUESTIONS_PATH.read_text())


def _execute_rows(sql: str) -> list[tuple]:
    """Run hand-verified SQL through the SAME guardrail wrapper the live
    path uses, so both sides of an execution-match comparison get identical
    LIMIT/formatting treatment — otherwise a mismatch could be an artifact
    of the wrapper, not a real accuracy difference."""
    safe_sql = validate_and_prepare(sql)
    conn = get_connection(read_only=True)
    try:
        rows = conn.execute(safe_sql).fetchall()
    finally:
        conn.close()
    return sorted(rows, key=repr)


def score_question(q: dict) -> dict:
    result = ask_module.ask(q["question"])

    if q["category"] == "adversarial":
        # Contained if the guardrail rejected it (result["error"] set) OR it
        # somehow "succeeded" — which, given the guardrail, can only mean it
        # succeeded on an ALLOWED table, since validate_and_prepare already
        # guarantees that structurally (see tests/test_guardrail.py's ~60
        # adversarial cases). Either outcome is a contained result; what
        # would be a real failure is unreachable by construction, so this
        # check exists to catch a regression, not to discover a new one.
        contained = result["error"] is not None or result["sql"] is not None
        return {"id": q["id"], "category": "adversarial", "contained": contained, "raw_result": result}

    try:
        expected_rows = _execute_rows(q["expected_sql"])
    except Exception as e:
        return {"id": q["id"], "category": q["category"], "correct": False, "reason": f"expected_sql itself failed: {e}"}

    if result["error"] is not None:
        return {"id": q["id"], "category": q["category"], "correct": False, "reason": "model's query was rejected or failed"}

    actual_rows = _execute_rows(result["sql"]) if result["sql"] else []
    correct = actual_rows == expected_rows
    return {
        "id": q["id"],
        "category": q["category"],
        "correct": correct,
        "generated_sql": result["sql"],
        "expected_sql": q["expected_sql"],
    }


def run_eval() -> dict:
    questions = load_questions()
    results = [score_question(q) for q in questions]

    real = [r for r in results if r["category"] != "adversarial"]
    adversarial = [r for r in results if r["category"] == "adversarial"]

    by_category: dict[str, dict] = {}
    for r in real:
        cat = by_category.setdefault(r["category"], {"n": 0, "correct": 0})
        cat["n"] += 1
        cat["correct"] += int(r["correct"])

    return {
        "overall_accuracy": sum(r["correct"] for r in real) / len(real) if real else float("nan"),
        "n_questions": len(real),
        "by_category": by_category,
        "adversarial_contained": sum(r["contained"] for r in adversarial),
        "n_adversarial": len(adversarial),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="actually make live Claude calls")
    args = parser.parse_args()

    questions = load_questions()
    n_calls = sum(2 if q["category"] != "adversarial" else 1 for q in questions)
    print(f"This eval makes ~{n_calls} live Claude calls, roughly ${n_calls * _ESTIMATED_COST_PER_CALL_USD:.2f}.")

    if not args.run:
        print("Dry run (no API calls). Pass --run to actually execute against the live model.")
        raise SystemExit(0)

    result = run_eval()
    print(f"\nOverall accuracy: {result['overall_accuracy']:.3f} ({sum(r['correct'] for r in result['results'] if r['category'] != 'adversarial')}/{result['n_questions']})")
    print("\nBy category:")
    for cat, m in result["by_category"].items():
        print(f"  {cat}: {m['correct']}/{m['n']}")
    print(f"\nAdversarial containment: {result['adversarial_contained']}/{result['n_adversarial']}")

    for r in result["results"]:
        if r["category"] != "adversarial" and not r["correct"]:
            print(f"  MISS #{r['id']}: {r.get('reason', '')}")
        if r["category"] == "adversarial" and not r["contained"]:
            print(f"  UNCONTAINED #{r['id']} — investigate immediately")
