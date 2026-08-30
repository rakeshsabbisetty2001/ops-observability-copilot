import datetime
import logging

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.db import get_connection
from app.middleware.body_limit import BodySizeLimitMiddleware
from app.middleware.rate_limit import API_RATE_LIMIT, _client_ip, limiter
from app.nl2sql.ask import ask

logger = logging.getLogger("ops_copilot.api")

app = FastAPI(title="AI Ops Observability Copilot")
app.state.limiter = limiter
app.add_middleware(BodySizeLimitMiddleware)

# The React frontend (web/) calls this API via browser `fetch`, unlike the old
# Streamlit UI which called it server-side and never triggered a CORS check.


def cors_origins(frontend_origin: str) -> list[str]:
    # Pulled out as its own function so the FRONTEND_ORIGIN branch is
    # testable directly against an arbitrary input, without reloading this
    # module (reload desyncs the app.config.settings singleton that other
    # test files' monkeypatch.setattr(config_module.settings, ...) calls
    # depend on being the SAME object app.db reads from — verified this
    # broke 8 unrelated tests in test_write_to_db.py/test_run_detector.py
    # when tried; see tests/test_cors.py, review round 1 finding #15).
    # Always includes the Vite dev default so local frontend dev works
    # without env setup; the deployed origin comes from settings, not
    # hardcoded (Render env var, same pattern as ANTHROPIC_MODEL etc.).
    # config.py's own validator already strips a trailing slash from
    # frontend_origin before this ever runs (review round 1, finding #2) —
    # this function trusts that, rather than stripping again.
    origins = ["http://localhost:5173"]
    if frontend_origin:
        origins.append(frontend_origin)
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(settings.frontend_origin),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # In the message string, not `extra={...}` — `extra` only attaches
    # LogRecord attributes, it does not put them in the rendered output
    # unless a formatter references them by name, and this app configures
    # none (falls through to logging.lastResort, whose format is the bare
    # message). The `extra=` version shipped in round 1 and never appeared
    # in a single log line, verified against the real container's own
    # `docker logs` (Epic 8 review round 2, Medium #2). app/middleware/
    # rate_limit.py's own docstring says the rightmost-XFF choice can't be
    # verified without a real deployment; this is what makes that
    # verifiable post-deploy instead of asserted — render.yaml ships
    # TRUST_PROXY=true before any such check exists. Read one of these
    # lines after the first deploy — if every client collapses onto the
    # same `client` value, Render is prepending rather than appending and
    # rate_limit.py's `[-1]` needs to become `[0]`.
    logger.warning("rate limit exceeded client=%s xff=%s", _client_ip(request), request.headers.get("x-forwarded-for"))
    return JSONResponse(status_code=429, content={"detail": "Too many requests, slow down."},
                         headers={"Retry-After": "60"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default handler echoes the offending `input` value back in
    # the 422 body — for /ask that's the raw question text; for /anomalies
    # it's the raw query/path param. Whitelist loc/type/msg instead of
    # blacklisting `input` — two real bugs in the blacklist version: (a) a
    # user-raised ValueError/AssertionError (from a future @field_validator)
    # puts a live exception object in `ctx`, which plain json.dumps can't
    # serialize, turning a 422 into an unhandled 500 (caught by the
    # catch-all below, but as a validation bug wearing a server-error
    # costume); (b) that same ValueError's message can carry the raw value
    # in `msg`, past an `input`-only filter. No request model here has a
    # custom validator today (verified: no ValueError/AssertionError path
    # is reachable), but the whitelist closes both classes at once rather
    # than depending on that staying true (Epic 8 review round 1, Medium
    # #3). `msg` is intentionally NOT stripped — it's what lets a client
    # fix a malformed request, and pydantic's own built-in messages are
    # always safe; the house rule for a future validator is on AskRequest,
    # where someone adding one would actually be looking.
    #
    # jsonable_encoder is belt-and-braces, not load-bearing by itself: with
    # this whitelist, every surviving field (loc: tuple[str|int], type:
    # str, msg: str) already serializes fine with plain json.dumps —
    # verified directly. It guards against a future pydantic field ending
    # up in this dict that isn't JSON-safe, the same class of bug that
    # made `ctx` dangerous here (Epic 8 review round 2, nit N1).
    errors = [{"loc": e.get("loc"), "type": e.get("type"), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": errors}))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong."})


class AskRequest(BaseModel):
    # extra="forbid": reject a typo'd/extra field instead of silently
    # ignoring it — Epic 8 is the epic that froze the public contract
    # (Epic 8 review round 1, nit N2). A forbidden extra's `loc` in the
    # 422 response carries the client-supplied field NAME, not a value —
    # not the content-echo path the RequestValidationError handler's
    # whitelist (below) guards against.
    #
    # House rule for any @field_validator added to a model in this file:
    # never interpolate a submitted value into a raised ValueError/
    # AssertionError message (e.g. f"bad value: {v}") — the validation
    # handler's whitelist keeps `msg`, so a value put there DOES reach the
    # client (Epic 8 review round 2, nit N2 — this note previously lived
    # only inside the handler, not where someone adding a validator would
    # be looking).
    model_config = {"extra": "forbid"}
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    question: str
    sql: str | None
    answer: str | None
    row_count: int | None
    anomaly_ids: list[int]
    error: str | None = None


class AnomalySummary(BaseModel):
    id: int
    service: str
    metric_name: str
    start_ts: datetime.datetime
    end_ts: datetime.datetime
    method: str
    score: float


class EventPoint(BaseModel):
    ts: datetime.datetime
    value: float
    in_window: bool


class AnomalyDetail(AnomalySummary):
    events: list[EventPoint]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(API_RATE_LIMIT)
def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    return AskResponse(**ask(body.question))


# Matches nl2sql's DEFAULT_ROW_LIMIT — same "don't ship an unbounded query"
# rule applies here even though this path is fixed SQL, not text-to-SQL
# (Epic 7 review round 1, Low #4: this project's own detector history shows
# row count is a tuning parameter, not a constant — a bare 3-sigma threshold
# once produced 595 rows).
_ANOMALIES_MAX_LIMIT = 500

_ANOMALY_COLS = ["id", "service", "metric_name", "start_ts", "end_ts", "method", "score"]


@app.get("/anomalies", response_model=list[AnomalySummary])
def list_anomalies(
    service: str | None = None,
    metric: str | None = None,
    since: datetime.datetime | None = Query(
        default=None,
        description="Compared against start_ts, which is stored as naive wall-clock "
        "UTC. A timezone-aware value is converted to UTC first; a naive value is "
        "compared as-is (Epic 7 review round 2, nit N4).",
    ),
    limit: int = Query(default=200, ge=1, le=_ANOMALIES_MAX_LIMIT),
) -> list[AnomalySummary]:
    where = []
    params: list = []
    # Empty-string filters are treated as "not set", matching the UI's own
    # `if service_filter:` guard — `service=` on the wire otherwise built
    # `WHERE service = ''`, which can never match a row (Epic 7 review round
    # 1, nit N5).
    if service:
        where.append("service = ?")
        params.append(service)
    if metric:
        where.append("metric_name = ?")
        params.append(metric)
    if since is not None:
        # Normalize to naive UTC before comparing — every timestamp this API
        # stores/emits is naive wall-clock, but DuckDB silently converts an
        # aware `since` to UTC before comparing, so the same wall-clock
        # instant given with different offsets returned different rows
        # (Epic 7 review round 1, nit N4).
        if since.tzinfo is not None:
            since = since.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        where.append("start_ts >= ?")
        params.append(since)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    conn = get_connection(read_only=True)
    try:
        rows = conn.execute(
            f"SELECT id, service, metric_name, start_ts, end_ts, method, score "
            f"FROM detected_anomalies {clause} ORDER BY start_ts DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    finally:
        conn.close()
    return [AnomalySummary(**dict(zip(_ANOMALY_COLS, row))) for row in rows]


# How far past the flagged window to pull events for the drill-down chart —
# a bare `start_ts..end_ts` window renders as a 2-10 point line with no
# baseline to compare against; verified live that 5 of 17 real anomalies
# rendered as a flat 2-point segment and the anomaly itself was invisible
# (e.g. #16's in-window 60-69 range looked like ordinary noise without its
# ~40 baseline) (Epic 7 review round 1, Medium #3). Capped at 6h: an
# unbounded `max(window_width, _DRILLDOWN_MIN_PAD)` scales the payload
# linearly with the anomaly's own width and has no ceiling — a 2h05m real
# anomaly already returned 3x its point count, and a sustained_drift
# anomaly (this project injects them) could return thousands into one
# Altair chart (Epic 7 review round 2, Low #2).
_DRILLDOWN_MIN_PAD = datetime.timedelta(minutes=30)
_DRILLDOWN_MAX_PAD = datetime.timedelta(hours=6)


def _drilldown_pad(window_width: datetime.timedelta) -> datetime.timedelta:
    # Pulled out as its own function so the floor/ceiling can be unit-tested
    # directly — an end-to-end assertion on timestamps returned against a
    # real corpus is too noisy (grid snapping, corpus-edge truncation) to
    # pin the exact _DRILLDOWN_MIN_PAD value; shrinking that constant to
    # ~0 still leaves *some* out-of-window point for any real anomaly
    # (pad then just equals window_width), so an end-to-end "is there
    # context on both sides" check alone can't catch a weakened floor
    # (Epic 7 review round 2, nit N10).
    return min(max(window_width, _DRILLDOWN_MIN_PAD), _DRILLDOWN_MAX_PAD)


@app.get("/anomalies/{anomaly_id}", response_model=AnomalyDetail)
def get_anomaly(anomaly_id: int = Path(ge=0, le=2**31 - 1)) -> AnomalyDetail:
    conn = get_connection(read_only=True)
    try:
        row = conn.execute(
            "SELECT id, service, metric_name, start_ts, end_ts, method, score "
            "FROM detected_anomalies WHERE id = ?",
            [anomaly_id],
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="anomaly not found")
        anomaly = dict(zip(_ANOMALY_COLS, row))

        # Pad by at least the window's own width (or a 30min floor for
        # near-instant windows) so a two-point flagged window still comes
        # back with real context either side.
        pad = _drilldown_pad(anomaly["end_ts"] - anomaly["start_ts"])
        padded_start, padded_end = anomaly["start_ts"] - pad, anomaly["end_ts"] + pad

        # The raw window behind the flag (plus context), not just the
        # detector's sample ids — lets the UI chart what actually happened,
        # per the architecture's drill-down flow. `_DRILLDOWN_MAX_PAD` bounds
        # the PAD, not the window itself — a wide anomaly (this project
        # injects sustained_drift, which can span the whole corpus) still
        # drives an unbounded response on its own; verified live a
        # whole-corpus window returned 4,032 events/249KB with the pad cap
        # unchanged (Epic 7 review round 3, Low #2). LIMIT is the actual
        # payload bound; the pad cap is a separate, legitimate knob for
        # chart legibility once the payload is already bounded.
        event_rows = conn.execute(
            "SELECT ts, value FROM events WHERE service = ? AND metric_name = ? "
            "AND ts >= ? AND ts <= ? ORDER BY ts LIMIT 2000",
            [anomaly["service"], anomaly["metric_name"], padded_start, padded_end],
        ).fetchall()
    finally:
        conn.close()

    events = [
        EventPoint(ts=ts, value=value, in_window=anomaly["start_ts"] <= ts <= anomaly["end_ts"])
        for ts, value in event_rows
    ]
    return AnomalyDetail(**anomaly, events=events)
