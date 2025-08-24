from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Project-wide configuration.

    Primary DB is read from env var DATABASE_URL.
    Example: postgresql+asyncpg://postgres:postgres@localhost:5432/pkmn
    """

    DATABASE_URL: str = (
        os.getenv("DATABASE_URL")
        or "postgresql+asyncpg://postgres:postgres@localhost:5432/pkmn"
    )

    # Optional: read a local .env if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Compatibility shim: earlier code used `settings.database_url`
    @property
    def database_url(self) -> str:
        return self.DATABASE_URL


settings = Settings()
