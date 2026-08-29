"""Epic 8: body-size limit, rate limiting, validation-error stripping, and
the generic 500 handler. Middleware/handlers ported verbatim from Projects
1/3 — these tests confirm the wiring in THIS app's main.py, not the
middleware logic itself (already covered by its own project's history)."""
import re
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app

client = TestClient(app)


def test_dockerfile_cmd_pins_the_flags_the_rate_limiter_depends_on():
    # These three flags are each a single-token edit that reopens a real,
    # previously-shipped defect with no other test noticing — this is the
    # same "trust boundary has no test" shape as High #2, relocated from
    # rate_limit.py into the Dockerfile's CMD by round 1's own fix (Epic 8
    # review round 2, Medium #3).
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    # startswith("CMD [") not just "CMD" — a decoy line like "CMDX_..." would
    # otherwise match too (Epic 8 review round 3, nit N1).
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD ["))
    assert "--no-proxy-headers" in cmd_line  # High #1: uvicorn's own proxy_headers default
    # A substring check ("--workers 1" in ...) is satisfied by --workers 10/16/100 —
    # exactly the silent 10x-100x rate-limit widening this flag exists to
    # prevent (Epic 8 review round 3, Low #2). \b or a following quote pins
    # the token; the CMD line ends `--workers 1"]`.
    assert re.search(r"--workers 1(?!\d)", cmd_line)  # Low #7: the rate limiter's storage is in-process
    assert "exec uvicorn" in cmd_line  # Medium #4: SIGTERM must reach uvicorn, not `sh`


def _request(headers: dict[bytes, bytes], client_host: str = "9.9.9.9") -> Request:
    return Request({
        "type": "http",
        "headers": list(headers.items()),
        "client": (client_host, 1234),
    })


def test_forged_xff_is_ignored_when_trust_proxy_is_off(monkeypatch):
    # trust_proxy=False (the shipped default) must ignore a fully
    # client-controlled header — TestClient's ASGI stack has no
    # ProxyHeadersMiddleware and a fixed scope["client"], so it cannot
    # observe uvicorn's own default rewriting this before the app runs
    # (that's what --no-proxy-headers in the Dockerfile's CMD is for); this
    # test only pins _client_ip's own trust gate, which is the one line
    # that was previously untested — deleting it left all 196 tests green
    # (Epic 8 review round 1, High #2).
    import app.config as config_module
    from app.middleware.rate_limit import _client_ip
    monkeypatch.setattr(config_module.settings, "trust_proxy", False)
    req = _request({b"x-forwarded-for": b"1.2.3.4, 5.6.7.8"})
    assert _client_ip(req) == "9.9.9.9"  # the real connection, not the header


def test_xff_rightmost_is_used_when_trust_proxy_is_on(monkeypatch):
    # Pins the leftmost/rightmost choice itself — trust_proxy=True (only
    # set in render.yaml, behind a real proxy) reads the RIGHTMOST entry.
    import app.config as config_module
    from app.middleware.rate_limit import _client_ip
    monkeypatch.setattr(config_module.settings, "trust_proxy", True)
    req = _request({b"x-forwarded-for": b"1.2.3.4, 5.6.7.8"})
    assert _client_ip(req) == "5.6.7.8"


def test_oversized_body_with_honest_content_length_gets_413():
    resp = client.post(
        "/ask",
        content=b'{"question": "' + b"x" * 60_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_malformed_content_length_gets_400_not_500():
    resp = client.post(
        "/ask",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )
    assert resp.status_code == 400


def test_oversized_chunked_body_with_no_content_length_gets_413():
    """The header check alone doesn't catch this — a body streamed without
    Content-Length (or with a lying one) has to be caught by capping actual
    bytes read, not just by rejecting a declared size up front."""
    def chunks():
        yield b'{"question": "'
        yield b"x" * 60_000
        yield b'"}'

    resp = client.post("/ask", content=chunks(), headers={"Content-Type": "application/json"})
    assert resp.status_code == 413


def test_normal_small_body_still_reaches_validation():
    """A body under the size cap must still flow through to the endpoint
    normally — proves the buffered-and-replayed body isn't corrupted."""
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # Pydantic's min_length, not blocked by size middleware


def test_validation_error_does_not_echo_the_raw_input():
    # A too-long question would otherwise come back inside exc.errors()'s
    # `input` field, echoing the submitted text into the 422 body.
    resp = client.post("/ask", json={"question": "x" * 2000})
    assert resp.status_code == 422
    body = resp.json()
    assert "x" * 100 not in str(body)  # the offending value itself is gone
    assert "detail" in body
    for err in body["detail"]:
        assert "input" not in err  # only field path/message/type remain


def test_validation_handler_survives_a_value_error_with_a_live_exception_in_ctx():
    # The exact case Project 3's own code comment warned about: a
    # @field_validator raising ValueError(f"bad value: {v}") puts a LIVE
    # exception object in ctx["error"], which plain json.dumps can't
    # serialize — the blacklist-`input` version of this handler turned that
    # into an unhandled 500 instead of a 422 (Epic 8 review round 1, Medium
    # #3). No request model in this app has a custom validator today, so
    # this drives the handler directly against a real pydantic error rather
    # than needing a live route.
    import asyncio

    from pydantic import BaseModel, field_validator

    from app.main import validation_exception_handler

    class _Probe(BaseModel):
        q: str

        @field_validator("q")
        @classmethod
        def _check(cls, v):
            raise ValueError(f"bad value: {v}")

    try:
        _Probe(q="SECRET_PAYLOAD_12345")
        raise AssertionError("validator should have raised")
    except Exception as e:
        pydantic_error = e

    class _FakeRequestValidationError:
        def errors(self):
            return pydantic_error.errors()

    resp = asyncio.run(validation_exception_handler(None, _FakeRequestValidationError()))
    assert resp.status_code == 422  # not 500 — the handler must not crash on a live ctx["error"]
    body = resp.body.decode()
    assert '"ctx"' not in body  # the live exception object itself is gone
    assert '"input"' not in body
    # `msg` still carries the value here, because THIS validator interpolates
    # it — that's a house-rule violation in the validator, not something the
    # handler can generically strip without also stripping pydantic's own
    # safe built-in messages ("String should have at most 1000 characters").
    # No request model in this app does this today (verified separately in
    # test_validation_error_does_not_echo_the_raw_input); this test exists
    # so the crash (the actual Epic 8 defect) can never regress silently.
    assert "SECRET_PAYLOAD_12345" in body


def test_unhandled_exception_returns_generic_500(monkeypatch):
    # raise_server_exceptions=False: the default TestClient re-raises server
    # exceptions for debugging, which would make the catch-all handler
    # untestable — same reason Project 3's test_main.py uses a second
    # client (tests/test_main.py:13-17).
    client_no_raise = TestClient(app, raise_server_exceptions=False)

    def _boom(read_only):
        raise RuntimeError("SELECT * FROM query_log WHERE secret='leak-me'")

    monkeypatch.setattr("app.main.get_connection", _boom)
    resp = client_no_raise.get("/anomalies")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Something went wrong."}
    assert "leak-me" not in resp.text
    assert "query_log" not in resp.text


def test_rate_limit_returns_429_with_retry_after(monkeypatch, caplog):
    from app.middleware.rate_limit import limiter
    from app.nl2sql import ask as ask_module
    # Mock the two live-API calls so this exercises the rate limiter itself,
    # not a real Anthropic round trip 10+ times.
    monkeypatch.setattr(ask_module, "generate_sql", lambda q: "SELECT 1")
    monkeypatch.setattr(ask_module, "summarize_result", lambda q, rows: "ok")

    limiter.reset()  # isolate from whatever other tests already consumed this minute's quota
    try:
        # settings.rate_limit_per_minute defaults to 10 (app/config.py) —
        # 15 requests must exceed it regardless of what other tests already sent.
        # A distinctive XFF value, sent on every request — asserting its
        # REAL VALUE lands in the log, not just the "xff=" label, closes the
        # gap where the format string could be reverted to a bare literal
        # ("client= xff=", no interpolation) and still pass a labels-only
        # check (Epic 8 review round 3, Low #3).
        with caplog.at_level("WARNING"):
            responses = [client.post("/ask", json={"question": "y" * 30},
                                      headers={"X-Forwarded-For": "203.0.113.7"}) for _ in range(15)]
        statuses = [r.status_code for r in responses]
        assert 429 in statuses
        limited = responses[statuses.index(429)]
        assert limited.headers.get("retry-after") == "60"
        # Asserted on the RENDERED text, not a LogRecord attribute — an
        # extra={...} kwarg attaches attributes that never reach the actual
        # log line without a formatter that references them by name, which
        # this app doesn't configure. Round 1's extra={"xff": ...} version
        # passed a test asserting on caplog.records[0].xff while emitting
        # nothing anywhere in a real deployment's actual logs (Epic 8 review
        # round 2, Medium #2).
        assert "xff=203.0.113.7" in caplog.text
        # client=testclient is the honest value under TestClient (its ASGI
        # scope hardcodes ("testclient", 50000), and trust_proxy=False here
        # means the forged XFF above is correctly ignored for _client_ip
        # itself) — this also documents why this test can't observe
        # uvicorn's own proxy-header rewriting, which is what the
        # Dockerfile's --no-proxy-headers flag exists for.
        assert "client=testclient" in caplog.text
    finally:
        limiter.reset()  # don't leave other tests starved


def test_extra_field_is_rejected_not_silently_ignored():
    # Epic 8 freezes the public contract — extra="forbid" on AskRequest
    # rejects a typo'd/extra field instead of silently accepting it
    # (Epic 8 review round 1, nit N2; untested until round 2's Low #4).
    resp = client.post("/ask", json={"question": "hi", "evil": "x"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail[0]["type"] == "extra_forbidden"
    assert detail[0]["loc"][-1] == "evil"  # the field NAME, not a value — not a content-echo path
