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

# Lazily-created singletons to avoid constructing multiple engines/sessionmakers.
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return a process-wide AsyncEngine (lazy init)."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, future=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return a process-wide async sessionmaker (lazy init)."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _sessionmaker


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Async context manager for a DB session:

        async with get_session() as s:
            await s.execute(...)
    """
    sm = get_sessionmaker()
    async with sm() as session:
        yield session


# Legacy attribute some modules imported directly
engine: AsyncEngine = get_engine()

__all__ = ["get_engine", "get_sessionmaker", "get_session", "engine", "AsyncSession"]

