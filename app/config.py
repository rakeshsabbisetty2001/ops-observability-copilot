"""App configuration. Loaded once from .env via pydantic-settings."""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    # Pinned per project convention (Projects 1/3) — never "latest".
    anthropic_model: str = "claude-sonnet-5"

    duckdb_path: str = str(Path(__file__).resolve().parent.parent / "data" / "ops.duckdb")

    @field_validator("anthropic_api_key", "anthropic_model", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        # Project 1 hit a real bug from a stray leading space pasted into .env —
        # pydantic-settings' own .env parsing strips it, but not every loader does.
        return v.strip() if isinstance(v, str) else v


settings = Settings()
