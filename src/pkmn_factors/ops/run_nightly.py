from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, cast

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from pkmn_factors.config import settings  # type: ignore[attr-defined]
from pkmn_factors.eval.backtest import run as run_backtest
from pkmn_factors.universe import load_universe


async def _backtest_one(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: str | None,
    sem: asyncio.Semaphore,
) -> None:
    """Run a single backtest for one card, with concurrency control."""
    async with sem:
        await run_backtest(
            card_key=card_key,
            model_version=model_version,
            horizon_days=horizon_days,
            persist=persist,
            note=note,
        )


async def _refresh_trades_daily_cagg(conn: AsyncConnection) -> None:
    """
    TimescaleDB requires the refresh procedure to run outside a txn.
    We flip the connection to AUTOCOMMIT and call the procedure.
    """
    # IMPORTANT: On this SQLAlchemy version, execution_options is async -> await it.
    conn_autocommit = cast(
        AsyncConnection,
        await conn.execution_options(isolation_level="AUTOCOMMIT"),
    )
    await conn_autocommit.exec_driver_sql(
        "CALL refresh_continuous_aggregate('public.trades_daily', NULL, NULL);"
    )


async def run_nightly(
    universe_keys: Iterable[str],
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: Optional[str],
    max_concurrency: int,
    refresh_cagg: bool,
) -> None:
    """Run nightly backtests for the given universe of card keys."""
    sem = asyncio.Semaphore(max_concurrency)

    # Launch backtest tasks concurrently (bounded by the semaphore)
    tasks = [
        asyncio.create_task(
            _backtest_one(
                k,
                model_version=model_version,
                horizon_days=horizon_days,
                persist=persist,
                note=note,
                sem=sem,
            )
        )
        for k in universe_keys
    ]
    if tasks:
        await asyncio.gather(*tasks)

    # Optionally refresh daily CAGG after backtests
    if refresh_cagg:
        engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
        try:
            async with engine.connect() as conn:  # connect() (not begin) -> no txn
                await _refresh_trades_daily_cagg(conn)
        finally:
            await engine.dispose()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nightly universe backtests")
    p.add_argument("--universe", type=Path, required=True, help="CSV of card_keys")
    p.add_argument("--model-version", required=True)
    p.add_argument("--horizon-days", type=int, default=90)
    p.add_argument("--persist", action="store_true", help="Persist metrics/results")
    p.add_argument("--note", default=None)
    p.add_argument("--max-concurrency", type=int, default=5)
    p.add_argument(
        "--refresh-cagg",
        action="store_true",
        help="Refresh trades_daily continuous aggregate after backtests",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    universe = load_universe(args.universe)
    asyncio.run(
        run_nightly(
            universe_keys=universe,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            persist=args.persist,
            note=(args.note or None),
            max_concurrency=args.max_concurrency,
            refresh_cagg=args.refresh_cagg,
        )
    )


if __name__ == "__main__":
    main()
