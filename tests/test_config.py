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


def test_frontend_origin_empty_by_default():
    assert Settings().frontend_origin == ""
