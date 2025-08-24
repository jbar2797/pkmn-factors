from __future__ import annotations

import asyncio
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from pkmn_factors.db.base import get_engine


async def _refresh_views(engine: AsyncEngine, views: Sequence[str]) -> None:
    async with engine.begin() as conn:
        for v in views:
            await conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {v};"))


async def run(refresh_mv: bool = False) -> None:
    engine = get_engine()
    if refresh_mv:
        await _refresh_views(engine, ["cards_signals_trades_daily_mv"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-mv", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(refresh_mv=args.refresh_mv))


if __name__ == "__main__":
    main()
