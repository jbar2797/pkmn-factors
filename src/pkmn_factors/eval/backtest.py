from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.dialects.postgresql import JSONB  # <-- NEW

from pkmn_factors.config import settings


# ---------- Utilities ----------


def _to_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def _daily_prices_from_trades(df: pd.DataFrame) -> pd.Series:
    """
    Input columns: timestamp (tz-aware), price (numeric/Decimal).
    Output: daily close prices as a float Series indexed by day, forward-filled.
    """
    if df.empty:
        return pd.Series(dtype=float)

    d = df.copy()
    d["timestamp"] = _to_datetime_utc(d["timestamp"])
    d["price"] = pd.to_numeric(d["price"], errors="coerce").astype(float)  # force float

    d = d.sort_values("timestamp")
    daily = (
        d.resample("1D", on="timestamp")
        .last()[["price"]]
        .dropna()
        .rename(columns={"price": "close"})
    )
    daily["close"] = daily["close"].ffill().astype(float)
    return daily["close"]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if not dd.empty else float("nan")


# ---------- Data access ----------


async def _engine() -> AsyncEngine:
    return create_async_engine(
        str(settings.DATABASE_URL), echo=False, pool_pre_ping=True
    )


async def load_trades(engine: AsyncEngine, card_key: str) -> pd.DataFrame:
    q = sa.text(
        """
        SELECT timestamp, price
        FROM trades
        WHERE card_key = :card
        ORDER BY timestamp
        """
    )
    async with engine.connect() as conn:
        res = await conn.execute(q, {"card": card_key})
        rows = res.fetchall()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])
    return pd.DataFrame(rows, columns=["timestamp", "price"])


async def load_signals(
    engine: AsyncEngine, card_key: str, model_version: Optional[str], horizon_days: int
) -> pd.DataFrame:
    params: Dict[str, Any] = {"card": card_key, "hz": horizon_days}
    where_mv = ""
    if model_version:
        where_mv = "AND model_version = :mv"
        params["mv"] = model_version

    q = sa.text(
        f"""
        SELECT asof_ts, action
        FROM signals
        WHERE card_key = :card
          AND horizon_days = :hz
          {where_mv}
        ORDER BY asof_ts
        """
    )
    async with engine.connect() as conn:
        res = await conn.execute(q, params)
        rows = res.fetchall()
    if not rows:
        return pd.DataFrame(columns=["asof_ts", "action"])
    return pd.DataFrame(rows, columns=["asof_ts", "action"])


# ---------- Backtest core ----------


@dataclass
class BTResult:
    period_start: Optional[pd.Timestamp]
    period_end: Optional[pd.Timestamp]
    n_trades: int
    n_signals: int
    win_rate: float
    avg_return: float
    cum_return: float
    volatility: float
    sharpe: float
    max_drawdown: float


def _positions_from_signals(idx: pd.DatetimeIndex, signals: pd.DataFrame) -> pd.Series:
    """
    Create a daily position series from signals (BUY=1, SELL=0, HOLD=prev).
    """
    if signals.empty:
        return pd.Series(0.0, index=idx, dtype=float)

    s = signals.copy()
    s["asof_ts"] = _to_datetime_utc(s["asof_ts"])
    s = s.set_index("asof_ts").sort_index()

    last = 0.0
    steps = []
    for t, row in s.iterrows():
        a = str(row["action"]).upper()
        if a == "BUY":
            last = 1.0
        elif a == "SELL":
            last = 0.0
        steps.append((t, last))

    sig_series = pd.Series(
        [v for _, v in steps], index=[t for t, _ in steps], dtype=float
    )
    pos = sig_series.reindex(idx, method="ffill").fillna(0.0).astype(float)
    return pos


def _bt(prices: pd.Series, signals: pd.DataFrame) -> BTResult:
    if prices.empty:
        return BTResult(None, None, 0, int(len(signals)), *(float("nan"),) * 6)

    daily_ret = prices.pct_change().fillna(0.0).astype(float)
    pos = _positions_from_signals(daily_ret.index, signals).astype(float)
    strat_ret = (pos.shift(1).fillna(0.0).astype(float)) * daily_ret
    equity = (1.0 + strat_ret).cumprod()

    nonzero = int((strat_ret != 0).sum())
    win_rate = (
        float((strat_ret > 0).sum()) / float(nonzero) if nonzero else float("nan")
    )
    avg_ret = float(strat_ret.mean()) if not strat_ret.empty else float("nan")
    cum_return = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    vol = (
        float(strat_ret.std() * np.sqrt(252.0)) if not strat_ret.empty else float("nan")
    )
    sharpe = (
        float((strat_ret.mean() * np.sqrt(252.0)) / strat_ret.std())
        if strat_ret.std()
        else float("nan")
    )
    mdd = _max_drawdown(equity)

    return BTResult(
        period_start=prices.index.min(),
        period_end=prices.index.max(),
        n_trades=int(prices.shape[0]),
        n_signals=int(len(signals)),
        win_rate=win_rate,
        avg_return=avg_ret,
        cum_return=cum_return,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=mdd,
    )


async def _write_metrics(
    engine: AsyncEngine,
    card_key: str,
    model_version: str,
    horizon_days: int,
    bt: BTResult,
    extra: Dict[str, Any],
) -> None:
    q = sa.text(
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
    ).bindparams(
        sa.bindparam("params", type_=JSONB)
    )  # <-- NEW: tell SQLAlchemy this is JSONB

    payload: Dict[str, Any] = {
        "card_key": card_key,
        "model_version": model_version,
        "horizon_days": int(horizon_days),
        "period_start": (
            None
            if bt.period_start is None
            else pd.Timestamp(bt.period_start).to_pydatetime()
        ),
        "period_end": (
            None
            if bt.period_end is None
            else pd.Timestamp(bt.period_end).to_pydatetime()
        ),
        "n_trades": bt.n_trades,
        "n_signals": bt.n_signals,
        "win_rate": bt.win_rate,
        "avg_return": bt.avg_return,
        "cum_return": bt.cum_return,
        "volatility": bt.volatility,
        "sharpe": bt.sharpe,
        "max_drawdown": bt.max_drawdown,
        "notes": extra.get("notes"),
        "params": extra,  # dict -> JSONB via bound type
    }

    async with engine.begin() as conn:
        await conn.execute(q, payload)


def _print_report(
    card_key: str, model_version: str, horizon_days: int, bt: BTResult
) -> None:
    def fmt(x: float, fmtstr: str) -> str:
        return fmtstr.format(x) if not np.isnan(x) else "NaN"

    print("\n=== Backtest Report ===")
    print(f"Card:          {card_key}")
    print(f"Model version: {model_version}")
    print(f"Horizon days:  {horizon_days}")
    print(f"Period:        {bt.period_start} -> {bt.period_end}")
    print(f"Trades:        {bt.n_trades}")
    print(f"Signals:       {bt.n_signals}")
    print(f"Win rate:      {fmt(bt.win_rate, '{:.3f}')}")
    print(f"Avg return:    {fmt(bt.avg_return, '{:.6f}')}")
    print(f"Cum return:    {fmt(bt.cum_return, '{:.4f}')}")
    print(f"Volatility:    {fmt(bt.volatility, '{:.4f}')}")
    print(f"Sharpe:        {fmt(bt.sharpe, '{:.3f}')}")
    print(f"Max drawdown:  {fmt(bt.max_drawdown, '{:.4f}')}")
    print("=======================\n")


# ---------- CLI ----------


async def run(
    card_key: str,
    model_version: Optional[str],
    horizon_days: int,
    persist: bool,
    note: Optional[str],
) -> None:
    engine = await _engine()

    trades_df = await load_trades(engine, card_key)
    signals_df = await load_signals(
        engine, card_key, model_version=model_version, horizon_days=horizon_days
    )

    prices = _daily_prices_from_trades(trades_df)
    bt = _bt(prices, signals_df)

    mv = model_version or "latest"
    _print_report(card_key, mv, horizon_days, bt)

    if persist:
        extra = {
            "notes": note,
            "counts": {"prices_days": int(prices.shape[0])},
            "filters": {
                "card_key": card_key,
                "model_version": model_version,
                "horizon_days": horizon_days,
            },
        }
        await _write_metrics(engine, card_key, mv, horizon_days, bt, extra)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest signals vs trades and write metrics"
    )
    ap.add_argument(
        "--card-key",
        required=True,
        help="Card key (matches trades.card_key and signals.card_key)",
    )
    ap.add_argument(
        "--model-version",
        default=None,
        help="Filter to a specific model_version (default: all/latest)",
    )
    ap.add_argument(
        "--horizon-days",
        type=int,
        default=90,
        help="Signal horizon in days (default: 90)",
    )
    ap.add_argument("--persist", action="store_true", help="Write metrics row to DB")
    ap.add_argument(
        "--note", default=None, help="Optional note stored with the metrics row"
    )

    args = ap.parse_args()
    asyncio.run(
        run(
            card_key=args.card_key,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            persist=args.persist,
            note=args.note,
        )
    )


if __name__ == "__main__":
    main()
