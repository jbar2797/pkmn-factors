from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
import sqlalchemy as sa
from pandas import Series
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from pkmn_factors.config import settings


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    period_start: datetime
    period_end: datetime
    n_trades: int
    n_signals: int
    win_rate: float
    avg_return: float
    cum_return: float
    volatility: float
    sharpe: float
    max_drawdown: float


# --------------------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------------------


def _engine() -> AsyncEngine:
    # Some stubs don’t know about this attribute; cast keeps mypy happy.
    db_url = cast(str, getattr(settings, "database_url"))
    return create_async_engine(db_url, pool_pre_ping=True, future=True)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


async def _load_prices(engine: AsyncEngine, card_key: str) -> pd.DataFrame:
    """
    Return raw trades for the card as a DataFrame with columns:
      timestamp (tz-aware), price (float)
    """
    sql = sa.text(
        """
        SELECT asof_ts AS timestamp, price
        FROM trades
        WHERE card_key = :card_key
        ORDER BY asof_ts ASC
        """
    )
    async with engine.begin() as conn:
        rows = (await conn.execute(sql, {"card_key": card_key})).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # enforce float (handles Decimal)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df = df.dropna(subset=["timestamp", "price"])
    return df


async def _load_signals(
    engine: AsyncEngine, card_key: str, model_version: str
) -> pd.DataFrame:
    """
    Return signals (asof_ts, action) for the given card/model_version, sorted ASC.
    action is expected in {'BUY','SELL','HOLD'}.
    """
    sql = sa.text(
        """
        SELECT asof_ts, action
        FROM signals
        WHERE card_key = :card_key
          AND model_version = :model_version
        ORDER BY asof_ts ASC
        """
    )
    async with engine.begin() as conn:
        rows = (
            (
                await conn.execute(
                    sql, {"card_key": card_key, "model_version": model_version}
                )
            )
            .mappings()
            .all()
        )

    if not rows:
        return pd.DataFrame(columns=["asof_ts", "action"])

    df = pd.DataFrame(rows)
    df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)
    df["action"] = df["action"].astype(str)
    return df


# --------------------------------------------------------------------------------------
# Backtest core
# --------------------------------------------------------------------------------------


def _daily_last_series(df: pd.DataFrame) -> Series:
    """
    From raw trades (timestamp, price), produce a daily last-observed price series.
    """
    if df.empty:
        return pd.Series(dtype=float)
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["price"] = pd.to_numeric(d["price"], errors="coerce").astype(float)
    daily = d.resample("1D", on="timestamp")["price"].last().dropna()

    # Ensure a DatetimeIndex (UTC) for downstream ops and type-checkers
    idx = pd.DatetimeIndex(daily.index)
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    daily.index = idx

    return daily.astype(float)


def _positions_from_signals(index: pd.DatetimeIndex, signals: pd.DataFrame) -> Series:
    """
    Create a position series (1 = long, -1 = short, 0 = flat) aligned to `index`.
    Positions jump at/after each signal time.
    """
    pos = pd.Series(index=index, dtype=float, data=0.0)
    if signals.empty:
        return pos

    def _act_to_pos(action: str) -> float:
        a = action.upper()
        if a == "BUY":
            return 1.0
        if a == "SELL":
            return -1.0
        return 0.0

    for row in signals.itertuples(index=False):
        # typing: make pandas happy about inputs here
        ts = pd.to_datetime(getattr(row, "asof_ts"), utc=True)  # type: ignore[arg-type]
        loc = int(pos.index.searchsorted(ts))
        if loc < len(pos.index):
            pos.iloc[loc:] = _act_to_pos(str(getattr(row, "action")))
    return pos


def _max_drawdown_from_returns(ret: Series) -> float:
    """
    Compute max drawdown magnitude (positive number) from strategy returns.
    """
    if ret.empty:
        return float("nan")
    eq_curve: pd.Series = (1.0 + ret.fillna(0.0)).cumprod()
    peak = eq_curve.cummax()
    dd = ((eq_curve / peak) - 1.0).min()
    return float(abs(dd))


def _bt(prices: Series, signals: pd.DataFrame) -> BacktestResult:
    """
    Naive daily strategy:
      - positions determined by last known signal (BUY=+1, SELL=-1, HOLD=0)
      - strategy return = position_{t-1} * daily_return
    """
    prices = pd.to_numeric(prices, errors="coerce").astype(float).dropna()
    if prices.empty:
        now = datetime.now(timezone.utc)
        return BacktestResult(
            period_start=now,
            period_end=now,
            n_trades=0,
            n_signals=0,
            win_rate=float("nan"),
            avg_return=0.0,
            cum_return=0.0,
            volatility=0.0,
            sharpe=float("nan"),
            max_drawdown=0.0,
        )

    daily_ret = prices.pct_change().fillna(0.0)

    # positions aligned to daily index
    pos = _positions_from_signals(pd.DatetimeIndex(prices.index), signals)

    # strategy returns
    strat_ret = pos.shift(1).fillna(0.0) * daily_ret

    # metrics
    idx = pd.DatetimeIndex(prices.index)
    period_start = idx[0].to_pydatetime()
    period_end = idx[-1].to_pydatetime()
    n_trades = int(prices.shape[0])
    n_signals = int(signals.shape[0])

    wins = (strat_ret > 0).sum()
    total_days = strat_ret.shape[0]
    win_rate = float(wins / total_days) if total_days > 0 else float("nan")

    avg_return = float(strat_ret.mean()) if total_days > 0 else 0.0
    vol = float(strat_ret.std(ddof=1)) if total_days > 1 else 0.0
    cum_return = float((1.0 + strat_ret).prod() - 1.0) if total_days > 0 else 0.0
    sharpe = float(avg_return / vol * np.sqrt(365.0)) if vol > 0 else float("nan")

    max_dd = _max_drawdown_from_returns(strat_ret)

    return BacktestResult(
        period_start=period_start,
        period_end=period_end,
        n_trades=n_trades,
        n_signals=n_signals,
        win_rate=win_rate,
        avg_return=avg_return,
        cum_return=cum_return,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


async def _write_metrics(
    engine: AsyncEngine,
    card_key: str,
    model_version: str,
    horizon_days: int,
    bt: BacktestResult,
    extra: Dict[str, object],
) -> None:
    """
    Insert one row into metrics (JSONB params via json.dumps).
    """
    sql = sa.text(
        """
        INSERT INTO metrics (
            card_key, model_version, horizon_days,
            period_start, period_end,
            n_trades, n_signals, win_rate, avg_return, cum_return,
            volatility, sharpe, max_drawdown,
            notes, params
        ) VALUES (
            :card_key, :model_version, :horizon_days,
            :period_start, :period_end,
            :n_trades, :n_signals, :win_rate, :avg_return, :cum_return,
            :volatility, :sharpe, :max_drawdown,
            :notes, :params
        )
        """
    )

    payload = {
        "card_key": card_key,
        "model_version": model_version,
        "horizon_days": horizon_days,
        "period_start": bt.period_start,
        "period_end": bt.period_end,
        "n_trades": bt.n_trades,
        "n_signals": bt.n_signals,
        "win_rate": bt.win_rate,
        "avg_return": bt.avg_return,
        "cum_return": bt.cum_return,
        "volatility": bt.volatility,
        "sharpe": bt.sharpe,
        "max_drawdown": bt.max_drawdown,
        "notes": str(extra.get("notes", "")),
        # IMPORTANT for asyncpg JSONB: pass a JSON string
        "params": json.dumps(extra),
    }

    async with engine.begin() as conn:
        await conn.execute(sql, payload)


# --------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------


async def run(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: Optional[str],
) -> BacktestResult:
    engine = _engine()

    prices_df = await _load_prices(engine, card_key)
    daily = _daily_last_series(prices_df)

    signals_df = await _load_signals(engine, card_key, model_version)

    bt = _bt(daily, signals_df)

    # print report
    print("\n=== Backtest Report ===")
    print(f"Card:          {card_key}")
    print(f"Model version: {model_version}")
    print(f"Horizon days:  {horizon_days}")
    print(
        f"Period:        {bt.period_start.isoformat()} -> {bt.period_end.isoformat()}"
    )
    print(f"Trades:        {bt.n_trades}")
    print(f"Signals:       {bt.n_signals}")
    print(f"Win rate:      {bt.win_rate}")
    print(f"Avg return:    {bt.avg_return:.6f}")
    print(f"Cum return:    {bt.cum_return:.4f}")
    print(f"Volatility:    {bt.volatility:.4f}")
    print(f"Sharpe:        {bt.sharpe}")
    print(f"Max drawdown:  {bt.max_drawdown:.4f}")
    print("=======================\n")

    if persist:
        extra: Dict[str, object] = {
            "notes": note or "",
            "counts": {"prices_days": int(daily.shape[0])},
            "filters": {
                "card_key": card_key,
                "model_version": model_version,
                "horizon_days": horizon_days,
            },
        }
        await _write_metrics(engine, card_key, model_version, horizon_days, bt, extra)

    return bt


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple backtest & metrics writer")
    ap.add_argument("--card-key", required=True)
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    asyncio.run(
        run(
            card_key=args.card_key,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            persist=args.persist,
            note=args.note or "",
        )
    )


if __name__ == "__main__":
    main()
