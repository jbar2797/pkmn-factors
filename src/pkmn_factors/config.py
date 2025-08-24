from __future__ import annotations

import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    # read from env at runtime; default "" keeps mypy happy
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", ""),
        alias="DATABASE_URL",
    )

    @property
    def database_url(self) -> str:
        # fail loudly at runtime if the variable is missing
        if not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL is not set. Example: "
                "postgresql+asyncpg://postgres:postgres@localhost/pkmn"
            )
        return self.DATABASE_URL


# single, importable settings object
settings: _Settings = _Settings()


class AppInfo(BaseModel):
    name: str = "pkmn-factors"
    version: str = "0.0.1"
