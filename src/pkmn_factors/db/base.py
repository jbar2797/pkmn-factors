# src/pkmn_factors/db/base.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from pkmn_factors.config import settings

# Single async engine for the app (Timescale/Postgres via asyncpg)
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,  # <-- use UPPER-CASE field from Settings
    echo=False,
    pool_pre_ping=True,
)

# Session factory
SessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager for a DB session."""
    async with SessionLocal() as session:
        yield session
