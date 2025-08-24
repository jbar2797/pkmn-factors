# src/pkmn_factors/config.py
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # General
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # Database (async SQLAlchemy URL for Postgres/Timescale via asyncpg)
    # If you run docker-compose from this repo, localhost:5432 is correct.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/pkmn"

    # Pydantic Settings config: load from .env if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton Settings instance (cached)."""
    return Settings()


# Convenient module-level instance
settings = get_settings()
