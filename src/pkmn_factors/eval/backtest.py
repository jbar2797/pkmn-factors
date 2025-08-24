from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pandas import Series
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# =============================================================================
# Types & dataclasses
# =============================================================================


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


# =============================================================================
# Helpers
# =============================================================================


def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return cast(pd.DatetimeIndex, idx.tz_convert("UTC"))


def _max_drawdown_from_returns(ret: Series) -> float:
    """
    ret: strategy daily returns (float series). Returns positive drawdown magnitude.
    """
    if ret.size == 0:
        return 0.0

    eq_curve: Series = (1.0 + ret.fillna(0.0)).cumprod()
    peak: Series = eq_curve.cummax()
    drawdown: Series = (eq_curve / peak) - 1.0
    dd = float(drawdown.min()) if drawdown.size else 0.0
    return abs(dd)


def _positions_from_signals(
    signals: pd.DataFrame, daily_idx: pd.DatetimeIndex
) -> Series:
    """
    Convert signals dataframe to a daily position series aligned to daily_idx.
    Rule: BUY => +1, SELL => -1, HOLD/None => 0.
    If multiple signals on a day, last one wins.
    """
    if signals.empty:
        return pd.Series(0.0, index=daily_idx, dtype=float)

    s = signals.copy()
    s["asof_ts"] = pd.to_datetime(s["asof_ts"], utc=True)
    s["day"] = s["asof_ts"].dt.tz_convert("UTC").dt.normalize()

    def _to_pos(a: str) -> float:
        a_up = (a or "").upper()
        if a_up == "BUY":
            return 1.0
        if a_up == "SELL":
            return -1.0
        return 0.0

    s["pos"] = s["action"].map(_to_pos).astype(float)
    # last signal of each day
    s = s.sort_values("asof_ts").groupby("day").tail(1)

    pos = pd.Series(0.0, index=daily_idx, dtype=float)
    # align days in index space
    day_index = pd.DatetimeIndex(s["day"])
    in_idx = day_index.intersection(daily_idx)
    if not in_idx.empty:
        day_to_pos = s.set_index("day")["pos"]
        mapped: Series = day_to_pos.reindex(in_idx).astype(float)
        pos.loc[in_idx] = mapped.values
    # forward fill positions day-to-day
    pos = pos.ffill().fillna(0.0).astype(float)
    return pos


# =============================================================================
# DB loaders (robust to 'timestamp' vs 'ts' in trades)
# =============================================================================


async def _load_prices_for_card(session: AsyncSession, card_key: str) -> pd.DataFrame:
    """
    Load raw prices for a card from the `trades` table.

    Some databases may store the time column as 'timestamp' while older snapshots
    used 'ts'. This routine tries both, returning the first that works.
    """
    candidates: List[str] = ["timestamp", "ts"]

    for col in candidates:
        q = text(
            f"""
            SELECT {col} AS timestamp, price
            FROM trades
            WHERE card_key = :card_key
            ORDER BY {col}
            """
        )
        try:
            result = await session.execute(q, {"card_key": card_key})
            rows_raw = result.all()  # Sequence[Row]
            rows: List[Tuple[Any, Any]] = [(row[0], row[1]) for row in rows_raw]

            if not rows:
                return pd.DataFrame(columns=["timestamp", "price"])

            df = pd.DataFrame(rows, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
            return df

        except ProgrammingError:
            # column doesn't exist for this candidate; try next
            continue

        except Exception:
            # non-schema error — bubble up
            raise

    # none worked: return empty frame with expected schema
    return pd.DataFrame(columns=["timestamp", "price"])


async def _load_signals_for_card(
    session: AsyncSession, card_key: str, model_version: str, horizon_days: int
) -> pd.DataFrame:
    """
    Load signals for a card/model/horizon from the `signals` table.
    """
    q = text(
        """
        SELECT
            asof_ts,
            action,
            conviction,
            expected_return,
            risk,
            utility
        FROM signals
        WHERE card_key = :card_key
          AND model_version = :model_version
          AND horizon_days = :horizon_days
        ORDER BY asof_ts
        """
    )
    result = await session.execute(
        q,
        {
            "card_key": card_key,
            "model_version": model_version,
            "horizon_days": horizon_days,
        },
    )
    rows = result.all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "asof_ts",
                "action",
                "conviction",
                "expected_return",
                "risk",
                "utility",
            ]
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "asof_ts",
            "action",
            "conviction",
            "expected_return",
            "risk",
            "utility",
        ],
    )
    df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)
    return df


# =============================================================================
# Core backtest
# =============================================================================


def _bt(prices: pd.DataFrame, signals: pd.DataFrame) -> tuple[Backtest, Dict[str, Any]]:
    """
    Combine prices and signals into daily strategy returns and compute metrics.
    """
    # Normalize / daily last price
    if prices.empty:
        now = datetime.now(timezone.utc)
        empty = Backtest(
            period_start=now,
            period_end=now,
            n_trades=0,
            n_signals=int(signals.shape[0]),
            win_rate=float("nan"),
            avg_return=0.0,
            cum_return=0.0,
            volatility=0.0,
            sharpe=float("nan"),
            max_drawdown=0.0,
        )
        return empty, {"counts": {"prices_days": 0}}

    p = prices.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    p["price"] = pd.to_numeric(p["price"], errors="coerce").astype(float)
    daily: Series = p.resample("1D", on="timestamp")["price"].last().dropna()

    if daily.empty:
        now = datetime.now(timezone.utc)
        empty = Backtest(
            period_start=now,
            period_end=now,
            n_trades=0,
            n_signals=int(signals.shape[0]),
            win_rate=float("nan"),
            avg_return=0.0,
            cum_return=0.0,
            volatility=0.0,
            sharpe=float("nan"),
            max_drawdown=0.0,
        )
        return empty, {"counts": {"prices_days": 0}}

    # Ensure UTC index
    daily_idx: pd.DatetimeIndex = _ensure_utc_index(pd.DatetimeIndex(daily.index))
    daily_ret: Series = daily.pct_change().fillna(0.0).astype(float)

    # Positions and strategy returns
    pos: Series = _positions_from_signals(signals, daily_idx).astype(float)
    strat_ret: Series = pos.shift(1).fillna(0.0) * daily_ret

    # Period bounds
    period_start = cast(datetime, daily_idx[0].to_pydatetime())
    period_end = cast(datetime, daily_idx[-1].to_pydatetime())

    # Metrics (be explicit for mypy)
    avg_ret: float = float(strat_ret.mean()) if strat_ret.size else 0.0
    vol: float = float(strat_ret.std(ddof=1)) if strat_ret.size > 1 else 0.0
    sharpe: float = float(avg_ret / vol) if vol > 0 else float("nan")

    # ---- key change: make ndarray[float] explicit before np.prod / float()
    prod_series: Series = (1.0 + strat_ret).astype(float)
    prod_array = prod_series.to_numpy(dtype=float)
    cum_ret: float = (float(np.prod(prod_array)) - 1.0) if prod_array.size else 0.0

    mdd: float = _max_drawdown_from_returns(strat_ret)

    took_pos: Series = pos.shift(1).fillna(0.0) != 0.0
    wins: int = int((strat_ret[took_pos] > 0.0).sum())
    total: int = int(took_pos.sum())
    win_rate: float = float(wins / total) if total > 0 else float("nan")

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


# =============================================================================
# Persist to metrics (JSONB-safe)
# =============================================================================


async def _write_metrics(
    engine: AsyncEngine,
    card_key: str,
    model_version: str,
    horizon_days: int,
    bt: Backtest,
    extra: Dict[str, Any],
    note: Optional[str],
) -> None:
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
        "params": json.dumps(
            {
                "notes": note or "",
                "counts": extra.get("counts", {}),
                "filters": {
                    "card_key": card_key,
                    "model_version": model_version,
                    "horizon_days": horizon_days,
                },
            }
        ),
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
            :notes, :params
        )
        """
    )

    async with engine.begin() as conn:
        await conn.execute(q, payload)


# =============================================================================
# Orchestration (run/main)
# =============================================================================


async def run(
    card_key: str,
    model_version: str,
    horizon_days: int,
    persist: bool,
    note: Optional[str],
) -> None:
    from pkmn_factors.config import settings  # typed Settings with DATABASE_URL

    engine = create_async_engine(settings.DATABASE_URL, future=True)
    SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with SessionLocal() as session:
        prices = await _load_prices_for_card(session, card_key)
        signals = await _load_signals_for_card(
            session, card_key, model_version, horizon_days
        )

    bt, extra = _bt(prices, signals)
    _print_report(card_key, model_version, horizon_days, bt)

    if persist:
        await _write_metrics(
            engine, card_key, model_version, horizon_days, bt, extra, note
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
