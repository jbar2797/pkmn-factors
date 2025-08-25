# src/pkmn_factors/ops/run_nightly.py
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from pkmn_factors.config import settings  # type: ignore[attr-defined]
from pkmn_factors.eval.backtest import run as run_backtest
from pkmn_factors.universe import load_universe


async def _refresh_trades_daily_cagg(conn: AsyncConnection) -> None:
    """
    TimescaleDB continuous aggregates require autocommit; run the refresh outside
    a transaction by switching the connection's isolation level.
    """
    # IMPORTANT: execution_options(...) is async in our SQLAlchemy version -> await it.
    aconn: AsyncConnection = await conn.execution_options(isolation_level="AUTOCOMMIT")
    await aconn.exec_driver_sql(
        "CALL refresh_continuous_aggregate('public.trades_daily', NULL, NULL);"
    )


async def _backtest_one(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: str | None,
) -> None:
    await run_backtest(
        card_key=card_key,
        model_version=model_version,
        horizon_days=horizon_days,
        persist=persist,
        note=note,
    )


async def run_nightly(
    universe: Iterable[str],
    model_version: str,
    horizon_days: int,
    persist: bool,
    refresh_cagg: bool,
    max_concurrency: int,
    note: str | None,
) -> None:
    # Optionally refresh CAGG up front
    if refresh_cagg:
        engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
        async with engine.connect() as conn:  # connect() (not begin) -> no txn
            await _refresh_trades_daily_cagg(conn)

    sem = asyncio.Semaphore(max_concurrency)

    async def _wrapped(card: str) -> None:
        async with sem:
            await _backtest_one(
                card_key=card,
                model_version=model_version,
                horizon_days=horizon_days,
                persist=persist,
                note=note,
            )

    await asyncio.gather(*(_wrapped(c) for c in universe))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nightly universe backtests")
    p.add_argument("--universe", type=Path, required=True, help="CSV of card_keys")
    p.add_argument("--model-version", required=True)
    p.add_argument("--horizon-days", type=int, default=90)
    p.add_argument("--persist", action="store_true")
    p.add_argument("--refresh-cagg", action="store_true")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--note", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cards = load_universe(args.universe)
    asyncio.run(
        run_nightly(
            universe=cards,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            persist=args.persist,
            refresh_cagg=args.refresh_cagg,
            max_concurrency=args.max_concurrency,
            note=args.note,
        )
    )


if __name__ == "__main__":
    main()
