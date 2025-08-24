from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# settings.database_url is provided by your project config
from pkmn_factors.config import settings  # type: ignore[attr-defined]


# --------------------------- Data classes ---------------------------


@dataclass(frozen=True)
class Backtest:
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


# --------------------------- Helpers ---------------------------


def _to_float_series(s: pd.Series) -> pd.Series:
    """Ensure float dtype even if inputs are Decimal/strings/ints."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a UTC tz-aware DatetimeIndex."""
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return cast(pd.DatetimeIndex, idx.tz_convert("UTC"))


def _positions_from_signals(
    signals: pd.DataFrame, daily_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Map actions to positions and forward-fill onto the daily price index.
    BUY -> +1, SELL -> -1, HOLD/other -> 0
    """
    if signals.empty:
        return pd.Series(0.0, index=daily_index)

    s = signals.copy()
    s["asof_ts"] = pd.to_datetime(s["asof_ts"], utc=True)

    action_map: Dict[str, float] = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
    s["pos"] = s["action"].map(action_map).fillna(0.0).astype(float)

    ser = pd.Series(s["pos"].values, index=pd.DatetimeIndex(s["asof_ts"], tz="UTC"))
    pos = ser.reindex(daily_index).ffill().fillna(0.0)
    return pos.astype(float)


def _max_drawdown_from_returns(ret: pd.Series) -> float:
    """Compute max drawdown magnitude (positive number) from a returns series."""
    if ret.empty:
        return float("nan")
    eq_curve = cast(pd.Series, (1.0 + ret.fillna(0.0)).cumprod()).astype(float)
    peak = cast(pd.Series, eq_curve.cummax()).astype(float)
    dd = (eq_curve / peak) - 1.0
    return float(abs(dd.min())) if dd.size else float("nan")


def _print_report(
    card_key: str, model_version: str, horizon_days: int, bt: Backtest
) -> None:
    print("\n=== Backtest Report ===")
    print(f"Card:          {card_key}")
    print(f"Model version: {model_version}")
    print(f"Horizon days:  {horizon_days}")
    print(f"Period:        {bt.period_start} -> {bt.period_end}")
    print(f"Trades:        {bt.n_trades}")
    print(f"Signals:       {bt.n_signals}")
    print(f"Win rate:      {bt.win_rate}")
    print(f"Avg return:    {bt.avg_return:.6f}")
    print(f"Cum return:    {bt.cum_return:.4f}")
    print(f"Volatility:    {bt.volatility:.4f}")
    print(f"Sharpe:        {bt.sharpe}")
    print(f"Max drawdown:  {bt.max_drawdown:.4f}")
    print("=======================\n")


# --------------------------- DB access ---------------------------


async def _load_prices_for_card(session: AsyncSession, card_key: str) -> pd.DataFrame:
    """
    Load raw trade prices for a card. Expected columns: timestamp (timestamptz), price (numeric).
    """
    q = text(
        """
        SELECT timestamp, price
        FROM trades
        WHERE card_key = :card_key
        ORDER BY timestamp
        """
    )
    res = await session.execute(q, {"card_key": card_key})
    rows = res.fetchall()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = _to_float_series(df["price"])
    return df


async def _load_signals_for_card(
    session: AsyncSession, card_key: str, model_version: str, horizon_days: int
) -> pd.DataFrame:
    """
    Load signals for a card. Expected columns: asof_ts (timestamptz), action (text).
    """
    q = text(
        """
        SELECT asof_ts, action
        FROM signals
        WHERE card_key = :card_key
          AND model_version = :model_version
          AND horizon_days = :horizon_days
        ORDER BY asof_ts
        """
    )
    res = await session.execute(
        q,
        {
            "card_key": card_key,
            "model_version": model_version,
            "horizon_days": horizon_days,
        },
    )
    rows = res.fetchall()
    if not rows:
        return pd.DataFrame(columns=["asof_ts", "action"])
    df = pd.DataFrame(rows, columns=["asof_ts", "action"])
    df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)
    return df


# --------------------------- Core backtest ---------------------------


def _bt(prices: pd.DataFrame, signals: pd.DataFrame) -> Tuple[Backtest, Dict[str, Any]]:
    """
    Very simple daily long/flat/-short backtest driven by signals.
    Returns (Backtest, extra_params_dict).
    """
    # Daily close price series
    daily = prices.resample("1D", on="timestamp")["price"].last().dropna()
    if daily.empty:
        now_utc = datetime.now(timezone.utc)
        empty = Backtest(
            period_start=now_utc,
            period_end=now_utc,
            n_trades=0,
            n_signals=0,
            win_rate=float("nan"),
            avg_return=0.0,
            cum_return=0.0,
            volatility=0.0,
            sharpe=float("nan"),
            max_drawdown=0.0,
        )
        return empty, {"counts": {"prices_days": 0}}

    # Ensure UTC index
    daily_idx = _ensure_utc_index(pd.DatetimeIndex(daily.index))
    daily_ret = daily.pct_change().fillna(0.0).astype(float)

    # Position series (shifted one day so today's return uses yesterday's signal)
    pos = _positions_from_signals(signals, daily_idx).astype(float)
    strat_ret = pos.shift(1).fillna(0.0) * daily_ret

    # Metrics (no datetime arithmetic with floats)
    period_start = cast(datetime, daily_idx[0].to_pydatetime())
    period_end = cast(datetime, daily_idx[-1].to_pydatetime())

    avg_ret = float(strat_ret.mean()) if strat_ret.size else 0.0
    vol = float(strat_ret.std(ddof=1)) if strat_ret.size > 1 else 0.0
    sharpe = float(avg_ret / vol) if vol > 0 else float("nan")
    cum_ret = float((1.0 + strat_ret).prod() - 1.0) if strat_ret.size else 0.0
    mdd = _max_drawdown_from_returns(strat_ret)

    took_pos = pos.shift(1).fillna(0.0) != 0.0
    wins = int((strat_ret[took_pos] > 0.0).sum())
    total = int(took_pos.sum())
    win_rate = float(wins / total) if total > 0 else float("nan")

    bt = Backtest(
        period_start=period_start,
        period_end=period_end,
        n_trades=int(daily.shape[0]),
        n_signals=int(signals.shape[0]),
        win_rate=win_rate,
        avg_return=avg_ret,
        cum_return=cum_ret,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=mdd,
    )
    return bt, {"counts": {"prices_days": int(daily.shape[0])}}


async def _write_metrics(
    engine: AsyncEngine,
    card_key: str,
    model_version: str,
    horizon_days: int,
    bt: Backtest,
    extra: Dict[str, Any],
    note: Optional[str] = None,
) -> None:
    """Persist one metrics row; params bound as JSONB via CAST."""
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
        "notes": note or "",
        "params": json.dumps(extra),
    }

    q = text(
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
            :notes, CAST(:params AS JSONB)
        )
        """
    )
    async with engine.begin() as conn:
        await conn.execute(q, payload)


# --------------------------- Orchestration / CLI ---------------------------


async def run(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: Optional[str],
) -> None:
    # Create engine and async session factory
    engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
    SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with SessionLocal() as session:
        prices = await _load_prices_for_card(session, card_key)
        signals = await _load_signals_for_card(
            session, card_key, model_version, horizon_days
        )

    bt, _extra = _bt(prices, signals)
    _print_report(card_key, model_version, horizon_days, bt)

    if persist:
        await _write_metrics(
            engine, card_key, model_version, horizon_days, bt, _extra, note
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--card-key", required=True)
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    asyncio.run(
        run(
            args.card_key,
            args.model_version,
            args.horizon_days,
            args.persist,
            args.note or None,
        )
    )


if __name__ == "__main__":
    main()
