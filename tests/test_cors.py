from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_allowed_origin_gets_cors_header():
    # Vite's dev-server default — always allowed regardless of FRONTEND_ORIGIN,
    # so local frontend dev never needs env setup (app/main.py).
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_unrelated_origin_gets_no_cors_header():
    # allow_origins is a real allowlist, not "*" — /ask is a live LLM call and
    # this is the only thing standing between it and any other site's JS.
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_preflight_allows_post_for_ask():
    r = client.options(
        "/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "POST" in r.headers.get("access-control-allow-methods", "")
