from app.config import Settings


def test_frontend_origin_trailing_slash_is_stripped():
    # A browser's Origin header is scheme+host+port with no trailing slash,
    # so a value pasted from a browser address bar (Vercel's own URL comes
    # with one) would otherwise match nothing and silently disable CORS
    # (review round 1, finding #2). Constructs Settings directly rather than
    # touching the app.config.settings singleton other tests monkeypatch.
    s = Settings(frontend_origin="https://ops-copilot.vercel.app/")
    assert s.frontend_origin == "https://ops-copilot.vercel.app"


def test_frontend_origin_without_trailing_slash_is_unchanged():
    s = Settings(frontend_origin="https://ops-copilot.vercel.app")
    assert s.frontend_origin == "https://ops-copilot.vercel.app"


def test_frontend_origin_empty_by_default(monkeypatch):
    # Settings() also reads the real process environment (on top of .env),
    # so without this the test would fail on any machine/CI runner that
    # happens to have FRONTEND_ORIGIN set — exactly the machine someone is
    # using to debug a CORS problem (review round 2, NEW-6). The other two
    # tests in this file are immune: init kwargs outrank both env sources in
    # pydantic-settings' default priority.
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    assert Settings(_env_file=None).frontend_origin == ""
