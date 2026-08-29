"""Ported from resume-screening-ai/app/middleware/rate_limit.py verbatim
(itself ported from sec-filings-rag/app/middleware/rate_limit.py)."""
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings


def _client_ip(request: Request) -> str:
    """Key the rate limit on the RIGHTMOST X-Forwarded-For entry — but only
    when settings.trust_proxy is on, meaning a real reverse proxy sits in
    front and appends the true client IP as the last hop. Without a proxy,
    X-Forwarded-For is entirely client-supplied and trusting it (rightmost
    or not) lets every request pick its own bucket by forging a new value —
    confirmed empirically in Projects 1/3. Falls back to the connection's
    actual remote address, which a client can't spoof.

    Rightmost is only correct if the deployed proxy topology has exactly
    one hop appending the real client IP — a claim this codebase can't
    verify without an actual deployment. Verify the real header shape
    post-deploy (log it once) before trusting this in production."""
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
API_RATE_LIMIT = f"{settings.rate_limit_per_minute}/minute"
