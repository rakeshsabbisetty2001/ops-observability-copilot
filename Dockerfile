FROM python:3.12-slim AS deps
WORKDIR /build
COPY requirements.txt .
# No embedding model / vector DB here (unlike sec-filings-rag) and no
# PDF/DOCX parsing (unlike resume-screening-ai) — text-to-SQL is a direct
# Claude structured-output call and detection runs offline, so a plain
# install is correct with no CPU/GPU torch concern and no model-bake step.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=deps /install /usr/local
COPY app/ app/
# data/ops.duckdb is pre-built and committed (the "commit the corpus"
# pattern, same as Project 1's Chroma index) — no live ingestion at build
# or request time. .dockerignore excludes data/query_log.duckdb (the app
# creates it fresh on first write) AND data/ground_truth.duckdb (the eval
# answer key — the API process never opens it, but there's no reason to
# ship it into the public image either; Epic 8 review round 1, Low #6).
COPY data/ data/

# Non-root user, same convention as Projects 1/3. Chown data/ specifically
# (not just create the user) — DuckDB needs to CREATE data/query_log.duckdb
# on first request and, per Project 1's Chroma lesson, a read_only=True
# connection can still need to touch a WAL/journal file even to open an
# existing one; COPY above ran as root, so without this the committed
# ops.duckdb would be unreadable to a non-root process.
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app/data
USER app
ENV HOME=/home/app

EXPOSE 8000
# Render (and most PaaS hosts) inject $PORT and expect the app to bind to
# it; hardcoding 8000 would boot successfully and be unreachable on deploy.
#
# --no-proxy-headers is NOT optional, unlike the comment this replaced
# claimed. uvicorn 0.32.1 defaults proxy_headers=True with
# forwarded_allow_ips="127.0.0.1" — omitting the flag does NOT leave
# proxy-header handling off. When the connecting peer is 127.0.0.1 (e.g.
# `uvicorn app.main:app` run directly, as 4 - Project Setup.md's own "Run
# it" section documents), uvicorn rewrites scope["client"] from a fully
# client-supplied X-Forwarded-For BEFORE the app ever runs — bypassing
# app/middleware/rate_limit.py's TRUST_PROXY gate one layer below itself.
# Verified live: stock `uvicorn app.main:app` let 30/30 forged-XFF requests
# through a 10/minute limit; the identical app with --no-proxy-headers
# correctly rate-limited them (Epic 8 review round 1, High #1). TRUST_PROXY
# is the intended single source of truth for whether X-Forwarded-For is
# trusted; --no-proxy-headers is what keeps that true.
#
# exec replaces `sh` as PID 1 with uvicorn, so it receives SIGTERM directly
# instead of `sh` absorbing it and Render/Docker force-killing after the
# grace period on every deploy/restart/spin-down (Epic 8 review round 1,
# Medium #4 — confirmed live: without exec, docker stop exits 137 with no
# shutdown log; with it, a clean shutdown and exit 0).
#
# --workers 1 (the uvicorn default, made explicit): slowapi's rate limiter
# is in-process memory, not shared across workers — a `--workers N` edit
# would silently turn the 10/minute limit into N×10/minute with no test,
# no log, and no error (Epic 8 review round 1, Low #7).
# ponytail: single-process only. Upgrade path if this ever needs to scale
# horizontally is Limiter(storage_uri="redis://...") in
# app/middleware/rate_limit.py, worth it only once real traffic justifies
# more than one worker.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers --workers 1"]
