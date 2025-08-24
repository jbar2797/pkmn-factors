from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from pkmn_factors.config import settings

# -----------------------
# central engine/factory
# -----------------------

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Return a singleton AsyncEngine constructed from settings.database_url."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, future=True)
    return _engine


def async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a singleton async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


# ---------------------------------------
# legacy compatibility (typed) aliases
# ---------------------------------------


def engine() -> AsyncEngine:
    """Back-compat alias: some modules import `engine` from here."""
    return get_engine()


@asynccontextmanager
async def get_session():
    """
    Back-compat alias: some modules do `async with get_session() as session:`.
    Yields an AsyncSession.
    """
    factory = async_session_maker()
    async with factory() as session:
        yield session
