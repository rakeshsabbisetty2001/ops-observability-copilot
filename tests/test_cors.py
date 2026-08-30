from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.main import app, cors_origins

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


def test_cors_origins_includes_dev_default_and_configured_frontend():
    # Direct test of the function the real app builds its allowlist from —
    # no reload, no shared-state risk (see app/main.py's comment on why:
    # reloading this module to exercise FRONTEND_ORIGIN broke 8 unrelated
    # tests elsewhere that monkeypatch app.config.settings, review round 1
    # finding #15).
    assert cors_origins("") == ["http://localhost:5173"]
    assert cors_origins("https://ops-copilot.vercel.app") == [
        "http://localhost:5173",
        "https://ops-copilot.vercel.app",
    ]


def test_configured_frontend_origin_is_actually_allowed_by_cors_middleware():
    # cors_origins() being right doesn't by itself prove the middleware
    # respects it — build a real CORSMiddleware-wrapped app from its output
    # and make a real request, same as the two tests above do against the
    # real app's dev origin.
    test_app = FastAPI()
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins("https://ops-copilot.vercel.app"),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @test_app.get("/ping")
    def ping():
        return {"ok": True}

    c = TestClient(test_app)
    r = c.get("/ping", headers={"Origin": "https://ops-copilot.vercel.app"})
    assert r.headers.get("access-control-allow-origin") == "https://ops-copilot.vercel.app"
    # The dev default must still work alongside a configured origin, not be
    # replaced by it.
    r = c.get("/ping", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
