from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceFeatures:
    ret_7: float
    ret_30: float
    risk: float
    drawdown_30: float
    value_z: float
    liq_7: float
    recency_days: float


def _safe_pct_change(s: pd.Series, periods: int) -> float:
    """Return percentage change over `periods`, robust to short or NaN series."""
    try:
        if len(s) <= periods or s.dropna().empty:
            return float("nan")
        end = s.iloc[-1]
        start = s.iloc[-periods - 1]
        if pd.isna(start) or start == 0:
            return float("nan")
        return float(end / start - 1.0)
    except Exception:
        return float("nan")


def _rolling_vol(s: pd.Series, window: int) -> float:
    try:
        r = s.pct_change().dropna()
        if r.empty:
            return float("nan")
        return float(r.tail(window).std(ddof=0))
    except Exception:
        return float("nan")


def _max_drawdown(s: pd.Series, window: int) -> float:
    """Max drawdown over a rolling window (as positive fraction)."""
    try:
        w = s.tail(window).astype(float)
        if w.empty:
            return float("nan")
        roll_max = w.cummax()
        dd = (w / roll_max) - 1.0
        return float(abs(dd.min()))
    except Exception:
        return float("nan")


def from_prices(df: pd.DataFrame) -> PriceFeatures:
    """
    Build baseline features from a price series dataframe with a 'price' column.
    Expects df.index to be datetime-like and increasing.
    """
    if "price" not in df.columns:
        raise ValueError("prices dataframe must contain a 'price' column")

    series = df["price"].astype(float)

    ret_7 = _safe_pct_change(series, 7)
    ret_30 = _safe_pct_change(series, 30)
    risk = _rolling_vol(series, 30)
    drawdown_30 = _max_drawdown(series, 30)

    # Value vs 30-day mean as a simple z-score proxy (mean-only, no variance normalization)
    try:
        ma30 = (
            float(series.tail(30).mean()) if not series.tail(30).empty else float("nan")
        )
        value_z = (
            float(series.iloc[-1] / ma30 - 1.0)
            if ma30 and not np.isnan(ma30)
            else float("nan")
        )
    except Exception:
        value_z = float("nan")

    # Liquidity proxy: number of observations in last 7 days
    try:
        liq_7 = float(series.last("7D").shape[0]) if not series.empty else 0.0
    except Exception:
        liq_7 = 0.0

    # Recency of last trade in days (negative if last timestamp is in the future)
    try:
        ts = pd.to_datetime(series.index)
        recency_days = float(
            (pd.Timestamp.utcnow(tz="UTC") - ts.max()).total_seconds() / 86400.0
        )
    except Exception:
        recency_days = float("nan")

    return PriceFeatures(
        ret_7=ret_7,
        ret_30=ret_30,
        risk=risk,
        drawdown_30=drawdown_30,
        value_z=value_z,
        liq_7=liq_7,
        recency_days=recency_days,
    )


def to_json(feats: PriceFeatures) -> Dict[str, float]:
    return {
        "ret_7": feats.ret_7,
        "ret_30": feats.ret_30,
        "risk": feats.risk,
        "drawdown_30": feats.drawdown_30,
        "value_z": feats.value_z,
        "liq_7": feats.liq_7,
        "recency_days": feats.recency_days,
    }
