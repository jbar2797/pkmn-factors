from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable, Optional, cast

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from pkmn_factors.config import settings  # type: ignore[attr-defined]
from pkmn_factors.eval.backtest import run as run_backtest
from pkmn_factors.universe import load_universe


async def _refresh_trades_daily_cagg(conn: AsyncConnection) -> None:
    """
    Timescale continuous-aggregate refresh must run outside a transaction.
    We set AUTOCOMMIT on the connection. SQLAlchemy's typing for
    `execution_options` is overly strict (coroutine), but at runtime it
    returns the connection synchronously. We cast to satisfy mypy.
    """
    aconn = cast(AsyncConnection, conn.execution_options(isolation_level="AUTOCOMMIT"))
    await aconn.exec_driver_sql(
        "CALL refresh_continuous_aggregate('public.trades_daily', NULL, NULL);"
    )


async def _backtest_one(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: Optional[str],
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        await run_backtest(
            card_key=card_key,
            model_version=model_version,
            horizon_days=horizon_days,
            persist=persist,
            note=note,
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
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(
            _backtest_one(k, model_version, horizon_days, persist, note, sem)
        )
        for k in universe_keys
    ]
    if tasks:
        await asyncio.gather(*tasks)

    if refresh_cagg:
        engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
        async with engine.connect() as conn:
            await _refresh_trades_daily_cagg(conn)
        await engine.dispose()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nightly backtest runner")
    p.add_argument(
        "--universe",
        default="data/universe_demo.csv",
        help="Path to CSV universe file (default: data/universe_demo.csv)",
    )
    p.add_argument(
        "--model-version",
        default="bhs_baseline_v2",
        help="Model version to use for all backtests",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=90,
        help="Backtest horizon in days (default: 90)",
    )
    p.add_argument(
        "--persist",
        action="store_true",
        help="Persist metrics to DB metrics table",
    )
    p.add_argument("--note", default="nightly", help="Optional note for runs")
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Max concurrent backtests (default: 5)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of universe items processed (0 = all)",
    )
    p.add_argument(
        "--refresh-cagg",
        action="store_true",
        help="Refresh Timescale trades_daily continuous aggregate after runs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print universe and exit without running backtests",
    )
    return p.parse_args()


def _load_universe(path: str, limit: int) -> list[str]:
    keys = sorted(load_universe(Path(path)))
    return keys[:limit] if limit and limit > 0 else keys


def main() -> None:
    args = _parse_args()
    keys = _load_universe(args.universe, args.limit)

    if args.dry_run:
        for k in keys:
            print(k)
        return

    asyncio.run(
        run_nightly(
            universe_keys=keys,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            persist=args.persist,
            note=args.note,
            max_concurrency=args.max_concurrency,
            refresh_cagg=args.refresh_cagg,
        )
    )


if __name__ == "__main__":
    main()
