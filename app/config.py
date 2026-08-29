"""App configuration. Loaded once from .env via pydantic-settings."""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    # Pinned per project convention (Projects 1/3) — never "latest".
    # render.yaml also pins ANTHROPIC_MODEL explicitly so the deployed
    # value is readable without cross-referencing this file — the two are
    # two sources of truth for the same default, kept in sync by comment
    # rather than by code (Epic 8 review round 2, nit N3).
    anthropic_model: str = "claude-sonnet-5"

    # Main DB: events, detected_anomalies, query_log — what the API reads/writes.
    duckdb_path: str = str(_REPO_ROOT / "data" / "ops.duckdb")
    # Separate file so ground truth is physically unreachable from the API's
    # connection, not just excluded by a guardrail regex (Epic 1-2 review #2).
    ground_truth_duckdb_path: str = str(_REPO_ROOT / "data" / "ground_truth.duckdb")
    # Also separate: DuckDB refuses to open a read_only connection to a file
    # that has ANY other connection open with a different read_only value,
    # regardless of matching config dicts (verified directly — this is not
    # fixable by aligning configs). The API's ops.duckdb connections must
    # stay read_only=True at all times; query_log needs to write on every
    # request, so it can't live in the same file (Epic 5 review round 1, #3).
    query_log_duckdb_path: str = str(_REPO_ROOT / "data" / "query_log.duckdb")

    # Production hardening (Epic 8) — same defaults/reasoning as Projects
    # 1/3, ported verbatim.
    rate_limit_per_minute: int = 10
    # /ask's question is capped at 1000 chars (app/main.py's AskRequest);
    # comfortably above that plus JSON escaping overhead.
    max_body_bytes: int = 50_000
    # False until deployed behind a real reverse proxy — with no proxy,
    # X-Forwarded-For is entirely client-supplied and trusting it would let
    # every request pick its own rate-limit bucket via a forged header
    # (verified in Projects 1/3). Flip to true only once deployed behind a
    # proxy that overwrites/appends this header itself.
    trust_proxy: bool = False

    @field_validator(
        "anthropic_api_key", "anthropic_model", "duckdb_path", "ground_truth_duckdb_path", "query_log_duckdb_path",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        # Project 1 hit a real bug from a stray leading space pasted into .env —
        # pydantic-settings' own .env parsing strips it, but not every loader does.
        return v.strip() if isinstance(v, str) else v

    @field_validator("duckdb_path", "ground_truth_duckdb_path", "query_log_duckdb_path", mode="after")
    @classmethod
    def resolve_relative_to_repo_root(cls, v: str) -> str:
        # A relative path (e.g. from .env) is otherwise cwd-dependent — run a
        # script from the wrong directory and DuckDB silently creates a fresh
        # empty file instead of erroring (Epic 1-2 review #8).
        p = Path(v)
        return str(p if p.is_absolute() else _REPO_ROOT / p)


settings = Settings()
