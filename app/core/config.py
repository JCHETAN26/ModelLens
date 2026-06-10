"""Application configuration, loaded from environment variables.

All tunables (DB URL, LLM provider, eval thresholds) are config-driven so the
system stays testable and secret-free by default.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Application
    app_env: str = "local"
    log_level: str = "INFO"

    # Database
    database_url: str = (
        "postgresql+psycopg://modellens:modellens@localhost:5432/modellens"
    )

    # LLM provider: "fake" (deterministic, no API key), "openai", or "anthropic".
    llm_provider: str = "fake"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Evaluation thresholds
    faithfulness_threshold: float = 0.85
    coverage_threshold: float = 0.75
    readability_max_grade: float = 12.0
    max_rewrite_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
