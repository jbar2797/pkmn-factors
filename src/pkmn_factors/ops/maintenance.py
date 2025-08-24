from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from pkmn_factors.db.base import get_engine


async def ping(engine: AsyncEngine) -> None:
    """Simple connectivity check."""
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))


async def main() -> None:
    engine = get_engine()
    await ping(engine)


if __name__ == "__main__":
    asyncio.run(main())
