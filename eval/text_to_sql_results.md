# Text-to-SQL Eval Results

**Status: harness built and verified, live numbers not yet run.** No `ANTHROPIC_API_KEY` has been added to `.env` for this project — the same state Projects 1 and 3 were in before Roy added a spend-capped key. `python -m eval.eval_text_to_sql --run` is ready to go the moment one lands (estimated ~38 calls, ~$0.38 at current pricing).

## What's built

- `eval/questions.json` — 16 real questions (5 lookup, 6 aggregation, 5 anomaly-linked) with hand-verified `expected_sql`, plus 6 adversarial questions covering the exact exploit classes Epic 5's review found and closed: non-SELECT injection, ground-truth exfiltration, `query_log` exfiltration, multi-statement injection, catalog dumping, and `SUMMARIZE`-as-data-source.
- `eval/eval_text_to_sql.py` — scores real questions by **execution match** (do the generated and expected SQL return the same rows, run through the identical guardrail-wrapped path so neither side gets an unfair LIMIT/formatting advantage), not string similarity. Scores adversarial questions by **containment**: did the guardrail block it, or — if the model "succeeded" — is that success necessarily on an allowed table (which the guardrail already guarantees structurally; see Epic 5's ~60 adversarial guardrail tests). Prints a pre-flight cost estimate before spending anything.
- `tests/test_eval_text_to_sql.py` — 8 tests exercise the harness's own scoring logic with a monkeypatched `generate_sql`: execution-match correctness (including a differently-worded-but-equivalent query correctly scoring as correct, not just an exact string match), incorrect-result detection, rejected-query handling, and both adversarial outcomes (guardrail rejects it outright; model complies fully with a jailbreak attempt and the guardrail still catches the literal malicious SQL). All pass — this is testing "does the eval measure correctly", not "is the model accurate", which is what the live run will answer.

## Why containment isn't really an open question

Epic 5's guardrail review (3 rounds, adversarial) already established structurally that no SQL reaching the database can touch a table outside `{events, detected_anomalies}` — verified against ~90+ hand-built adversarial SQL strings executed live against a seeded confidential record, zero leaks, across three rounds of a different reviewer trying to break it each time. This eval's adversarial subset measures something different and still worth knowing: **whether the six adversarial questions above actually provoke the model into generating disallowed SQL at all**, versus the model just declining in natural language. Either outcome is safe (the guardrail holds regardless), but it's a real, reportable finding about model behavior, not just guardrail behavior.

## Once a key lands

Running `--run` will populate:
- Overall execution-match accuracy, broken down by category (lookup / aggregation / anomaly-linked) — expect aggregation questions to be the hardest (`GROUP BY` + `date_trunc` phrasing has more ways to go subtly wrong than a `COUNT(*)`).
- Adversarial containment count (expected: 6/6, since the guardrail already guarantees this independent of model behavior) — a value under 6/6 here would mean this eval harness itself has a bug, not that data leaked, and should be treated as an eval bug to investigate before anything else.
