# mypy: disable-error-code=attr-defined
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pkmn_factors.config import settings
from pkmn_factors.db.models import Trade, Signal

# -----------------------------------------------------------------------------
# Config for the baseline
# -----------------------------------------------------------------------------

HORIZON_DAYS = 90
MODEL_VERSION = "1"  # VARCHAR in DB, so keep it as a string

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _pct_change(series: pd.Series, periods: int) -> float:
    """Return percentage change over `periods` days."""
    if len(series) < periods + 1:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-(periods + 1)] - 1.0)


def _drawdown(series: pd.Series, window: int) -> float:
    """Max drawdown over a trailing window (negative number)."""
    if series.empty:
        return np.nan
    roll_max = series.rolling(window=window, min_periods=1).max()
    dd = series / roll_max - 1.0
    return float(dd.tail(window).min())


# -----------------------------------------------------------------------------
# Decision container
# -----------------------------------------------------------------------------


@dataclass
class Decision:
    action: str  # "BUY" | "HOLD" | "SELL"
    conviction: float
    expected_return: float
    risk: float
    utility: float
    features: Dict[str, Any]


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


async def _load_trades(session: AsyncSession, card_key: str) -> pd.DataFrame:
    """Fetch trades for one card and coerce to float prices & UTC index."""
    q = (
        select(Trade.timestamp, Trade.price)
        .where(Trade.card_key == card_key)
        .order_by(Trade.timestamp)
    )
    rows = (await session.execute(q)).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])

    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    # Make timestamps tz-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # *** Important: prices come back as Decimal -> coerce to float ***
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)

    # Drop any nulls just in case and index by time
    df = df.dropna(subset=["price"]).set_index("timestamp").sort_index()
    return df


def _features_from_prices(prices: pd.Series) -> Dict[str, Any]:
    """Compute simple features out of a price series."""
    # Daily last price to make horizons consistent
    daily = prices.resample("1D").last().ffill()

    ret_7 = _pct_change(daily, 7)
    ret_30 = _pct_change(daily, 30)

    # Risk proxy: 30-day std of daily returns
    daily_ret = daily.pct_change().dropna()
    risk = float(daily_ret.tail(30).std()) if not daily_ret.empty else np.nan

    # Trailing drawdown
    dd_30 = _drawdown(daily, 30)

    # "Value" vs 30-day moving average (z-score)
    if len(daily) >= 30:
        ma = daily.rolling(30).mean()
        std = daily.rolling(30).std()
        denom = std.iloc[-1]
        value_z = (
            float((daily.iloc[-1] - ma.iloc[-1]) / denom)
            if denom and not np.isnan(denom)
            else np.nan
        )
    else:
        value_z = np.nan

    # Liquidity proxy: count of raw trades in last 7 days
    liq_7 = int(prices.last("7D").shape[0]) if not prices.empty else 0

    # Recency (days since last raw trade)
    now_utc = pd.Timestamp.now(tz="UTC")
    recency_days = (
        float((now_utc - prices.index[-1]) / timedelta(days=1))
        if not prices.empty
        else np.nan
    )

    return {
        "ret_7": ret_7,
        "ret_30": ret_30,
        "risk": risk,
        "drawdown_30": dd_30,
        "value_z": value_z,
        "liq_7": liq_7,
        "recency_days": recency_days,
    }


def _score_to_decision(feats: Dict[str, Any]) -> Decision:
    """Toy Buy/Hold/Sell rule using expected return and risk."""
    exp_ret = feats.get("ret_30")
    if exp_ret is None or np.isnan(exp_ret):
        exp_ret = feats.get("ret_7")
    if exp_ret is None or np.isnan(exp_ret):
        exp_ret = 0.0

    risk = feats.get("risk")
    if risk is None or np.isnan(risk):
        risk = 0.0

    # Simple utility function
    utility = float(exp_ret - 4.0 * risk)

    # Map to action
    if utility > 0.02:
        action = "BUY"
    elif utility < -0.02:
        action = "SELL"
    else:
        action = "HOLD"

    # Conviction is a squashed utility to 0..1
    conv = 1 / (1 + np.exp(-utility * 50))

    return Decision(
        action=action,
        conviction=float(conv),
        expected_return=float(exp_ret),
        risk=float(risk),
        utility=float(utility),
        features=feats,
    )


async def run(card_key: str) -> Decision:
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as session:
        df = await _load_trades(session, card_key)
        if df.empty:
            dec = Decision("HOLD", 0.0, 0.0, 0.0, 0.0, {})
        else:
            feats = _features_from_prices(df["price"])
            dec = _score_to_decision(feats)

        await _write_signal(session, card_key, dec)
        return dec


async def _write_signal(session: AsyncSession, card_key: str, dec: Decision) -> None:
    row = Signal(
        card_key=card_key,
        horizon_days=HORIZON_DAYS,
        action=dec.action,
        conviction=dec.conviction,
        expected_return=dec.expected_return,
        risk=dec.risk,
        utility=dec.utility,
        model_version=str(MODEL_VERSION),  # keep as string
        features=dec.features,  # JSONB
    )
    session.add(row)
    await session.commit()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card-key", required=True)
    args = p.parse_args()

    dec = asyncio.run(run(args.card_key))
    print(dec)


if __name__ == "__main__":
    main()
