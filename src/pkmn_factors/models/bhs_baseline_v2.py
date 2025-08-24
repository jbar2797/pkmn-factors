from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pkmn_factors.config import settings
from pkmn_factors.db.models import Trade, Signal
from pkmn_factors.factors.features import build_features, Features

MODEL_VERSION = "bhs_baseline_v2"
DEFAULT_HORIZON_DAYS = 90

WIN_RETURN = 30
WIN_RISK = 30
WIN_REG = 30
LAMBDA_RISK = 4.0


@dataclass
class Decision:
    action: str
    conviction: float
    expected_return: float
    risk: float
    utility: float
    features: Dict[str, Any]


async def _load_trades(session: AsyncSession, card_key: str) -> pd.DataFrame:
    q = (
        select(Trade.timestamp, Trade.price)
        .where(Trade.card_key == card_key)
        .order_by(Trade.timestamp)
    )
    rows = (await session.execute(q)).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])

    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df = df.dropna(subset=["price"]).sort_values("timestamp").set_index("timestamp")
    return df


def _rolling_expected_return(daily: pd.Series, win: int = WIN_RETURN) -> float:
    if daily.size < 2:
        return float("nan")
    r = daily.pct_change().dropna()
    if r.empty:
        return float("nan")
    r_tail = r.tail(win)
    return float(r_tail.mean()) if not r_tail.empty else float("nan")


def _rolling_risk(daily: pd.Series, win: int = WIN_RISK) -> float:
    r = daily.pct_change().dropna().tail(win)
    return float(r.std(ddof=1)) if not r.empty else float("nan")


def _trend_slope_logprice(daily: pd.Series, win: int = WIN_REG) -> float:
    s = daily.dropna().tail(win)
    if s.size < 2:
        return float("nan")
    y = np.log(np.asarray(s.values, dtype=float))  # ensure ndarray[float]
    x = np.arange(y.size, dtype=float)
    try:
        slope = float(np.polyfit(x, y, 1)[0])
        return slope
    except Exception:
        return float("nan")


def _prob_buy_from_utility(
    expected_return: float, risk: float, lam: float = LAMBDA_RISK
) -> Tuple[float, float]:
    if np.isnan(expected_return):
        expected_return = 0.0
    if np.isnan(risk):
        risk = 0.0
    utility = float(expected_return - lam * risk)
    temp = 50.0
    prob = 1.0 / (1.0 + np.exp(-utility * temp))
    return float(prob), float(utility)


def _action_from_utility(
    utility: float, buy_thresh: float = 0.02, sell_thresh: float = -0.02
) -> str:
    if utility > buy_thresh:
        return "BUY"
    if utility < sell_thresh:
        return "SELL"
    return "HOLD"


def _expected_return_blend(daily: pd.Series) -> float:
    mu = _rolling_expected_return(daily, WIN_RETURN)
    slope = _trend_slope_logprice(daily, WIN_REG)
    if np.isnan(mu) and np.isnan(slope):
        return float("nan")
    if np.isnan(mu):
        return slope
    if np.isnan(slope):
        return mu
    return float(0.5 * mu + 0.5 * slope)


def _compute_signal_from_prices(raw: pd.DataFrame) -> Tuple[Decision, Features]:
    df_reset = raw.reset_index()
    feats_struct = build_features(df_reset)

    daily = raw["price"].resample("1D").last().ffill()

    exp_ret = _expected_return_blend(daily)
    risk = _rolling_risk(daily)
    prob_buy, utility = _prob_buy_from_utility(exp_ret, risk, LAMBDA_RISK)
    action = _action_from_utility(utility)

    dec = Decision(
        action=action,
        conviction=float(prob_buy),
        expected_return=float(exp_ret if not np.isnan(exp_ret) else 0.0),
        risk=float(risk if not np.isnan(risk) else 0.0),
        utility=float(utility),
        features=feats_struct.as_json(),
    )
    return dec, feats_struct


async def _write_signal(
    session: AsyncSession, card_key: str, dec: Decision, horizon_days: int
) -> None:
    row = Signal(
        card_key=card_key,
        horizon_days=horizon_days,
        action=dec.action,
        conviction=dec.conviction,
        expected_return=dec.expected_return,
        risk=dec.risk,
        utility=dec.utility,
        model_version=MODEL_VERSION,
        features=dec.features,
    )
    session.add(row)
    await session.commit()


async def run(card_key: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> Decision:
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as session:
        raw = await _load_trades(session, card_key)
        if raw.empty:
            empty = Decision("HOLD", 0.5, 0.0, 0.0, 0.0, {})
            await _write_signal(session, card_key, empty, horizon_days)
            return empty

        dec, _ = _compute_signal_from_prices(raw)
        await _write_signal(session, card_key, dec, horizon_days)
        return dec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card-key", required=True)
    p.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    args = p.parse_args()

    decision = asyncio.run(run(args.card_key, args.horizon_days))
    print(decision)


if __name__ == "__main__":
    main()
