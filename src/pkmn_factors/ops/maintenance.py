from __future__ import annotations

import asyncio
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from pkmn_factors.config import settings


def create_engine(
    url: str | None = None, echo: bool = False, pool_pre_ping: bool = True
) -> AsyncEngine:
    return create_async_engine(
        url or settings.DATABASE_URL, echo=echo, pool_pre_ping=pool_pre_ping
    )


async def _exec_many(engine: AsyncEngine, stmts: Sequence[sa.sql.Executable]) -> None:
    async with engine.begin() as conn:
        for s in stmts:
            await conn.execute(s)


async def refresh_trades_daily() -> None:
    engine = create_engine()
    await _exec_many(
        engine, [sa.text("REFRESH MATERIALIZED VIEW CONCURRENTLY trades_daily")]
    )


async def seed_cards() -> None:
    engine = create_engine()
    stmt = sa.text(
        """
        INSERT INTO cards (card_key, name, set_code, rarity)
        VALUES (:card_key, :name, :set_code, :rarity)
        ON CONFLICT (card_key) DO NOTHING
        """
    )
    params = [
        {
            "card_key": "mew-ex-053-svp-2023",
            "name": "Mew ex #053",
            "set_code": "SVP",
            "rarity": "Promo",
        },
    ]
    async with engine.begin() as conn:
        for p in params:
            await conn.execute(stmt, p)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-demo", action="store_true")
    ap.add_argument("--refresh-mv", action="store_true")
    args = ap.parse_args()

    if args.seed_demo:
        asyncio.run(seed_cards())
    if args.refresh_mv:
        asyncio.run(refresh_trades_daily())
