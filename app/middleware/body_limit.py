"""Reject oversized request bodies before they get buffered into memory.

Ported from resume-screening-ai/app/middleware/body_limit.py verbatim
(itself ported from sec-filings-rag) — same reasoning: Pydantic's
Field(max_length=...) only validates after Starlette has already read the
whole body into memory, and FastAPI's own body-parsing code swallows
anything raised while reading the body into a generic 400 regardless of
what status the raised exception intended (verified empirically in
Project 1). Deciding and responding directly in this middleware, before
any of that machinery runs, avoids depending on that.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings


def _too_large() -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": "Request body too large."})


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            if length > settings.max_body_bytes:
                return _too_large()

        chunks = []
        total = 0
        more_body = True
        while more_body:
            message = await request.receive()
            total += len(message.get("body", b""))
            if total > settings.max_body_bytes:
                return _too_large()
            chunks.append(message)
            more_body = message.get("more_body", False)

        async def replay_receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.disconnect"}

        request._receive = replay_receive
        return await call_next(request)
