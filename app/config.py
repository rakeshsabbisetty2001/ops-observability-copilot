"""App configuration. Loaded once from .env via pydantic-settings."""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    # Pinned per project convention (Projects 1/3) — never "latest".
    anthropic_model: str = "claude-sonnet-5"

    # Main DB: events, detected_anomalies, query_log — what the API reads/writes.
    duckdb_path: str = str(_REPO_ROOT / "data" / "ops.duckdb")
    # Separate file so ground truth is physically unreachable from the API's
    # connection, not just excluded by a guardrail regex (Epic 1-2 review #2).
    ground_truth_duckdb_path: str = str(_REPO_ROOT / "data" / "ground_truth.duckdb")

    @field_validator("anthropic_api_key", "anthropic_model", "duckdb_path", "ground_truth_duckdb_path", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        # Project 1 hit a real bug from a stray leading space pasted into .env —
        # pydantic-settings' own .env parsing strips it, but not every loader does.
        return v.strip() if isinstance(v, str) else v

    @field_validator("duckdb_path", "ground_truth_duckdb_path", mode="after")
    @classmethod
    def resolve_relative_to_repo_root(cls, v: str) -> str:
        # A relative path (e.g. from .env) is otherwise cwd-dependent — run a
        # script from the wrong directory and DuckDB silently creates a fresh
        # empty file instead of erroring (Epic 1-2 review #8).
        p = Path(v)
        return str(p if p.is_absolute() else _REPO_ROOT / p)


settings = Settings()
