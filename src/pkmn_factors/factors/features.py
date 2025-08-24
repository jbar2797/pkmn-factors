# src/pkmn_factors/factors/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from datetime import datetime, timezone


@dataclass(frozen=True)
class Features:
    ret_7: float
    ret_30: float
    risk: float
    drawdown_30: float
    value_z: float
    liq_7: float
    recency_days: float

    def as_json(self) -> Dict[str, float]:
        return {
            "ret_7": self.ret_7,
            "ret_30": self.ret_30,
            "risk": self.risk,
            "drawdown_30": self.drawdown_30,
            "value_z": self.value_z,
            "liq_7": self.liq_7,
            "recency_days": self.recency_days,
        }


def _to_float_series(s: pd.Series) -> pd.Series:
    """Ensure numeric series is float (Decimal -> float)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _safe_pct_change(s: pd.Series, periods: int) -> float:
    try:
        if len(s) <= periods or s.dropna().empty:
            return float("nan")
        end = float(s.iloc[-1])
        start = float(s.iloc[-periods - 1])
        if pd.isna(start) or start == 0:
            return float("nan")
        return float(end / start - 1.0)
    except Exception:
        return float("nan")


def _rolling_vol(s: pd.Series, window: int) -> float:
    try:
        v = s.pct_change().dropna().tail(window).std(ddof=1)
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def _max_drawdown(s: pd.Series, window: int) -> float:
    try:
        w = s.tail(window).astype(float).to_numpy()
        if w.size == 0:
            return float("nan")
        peak = -np.inf
        mdd = 0.0
        for x in w:
            peak = max(peak, x)
            if peak > 0:
                mdd = min(mdd, (x / peak) - 1.0)
        return float(mdd)
    except Exception:
        return float("nan")


def _zscore_latest(s: pd.Series, window: int) -> float:
    try:
        w = s.tail(window).astype(float)
        if w.empty:
            return float("nan")
        mu = float(w.mean())
        sd = float(w.std(ddof=1))
        if sd == 0 or pd.isna(sd):
            return float("nan")
        latest = float(w.iloc[-1])
        return float((latest - mu) / sd)
    except Exception:
        return float("nan")


def build_features(df: pd.DataFrame) -> Features:
    """
    df columns expected: timestamp (datetime64 tz-aware), price (numeric), card_key (str)
    """
    if df.empty:
        return Features(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    # enforce types
    prices = _to_float_series(df["price"])
    daily = prices.resample("1D", on="timestamp").last().dropna()

    ret_7 = _safe_pct_change(daily, 7)
    ret_30 = _safe_pct_change(daily, 30)
    risk = _rolling_vol(daily, 30)
    drawdown_30 = _max_drawdown(daily, 30)
    value_z = _zscore_latest(daily, 30)

    # liquidity = number of trades in last 7 days
    liq_7 = float(
        (df.set_index("timestamp").last("7D").shape[0]) if not df.empty else 0
    )

    # recency in days from last trade to now (UTC)
    last_ts = pd.to_datetime(df["timestamp"].max(), utc=True)
    now_utc = datetime.now(timezone.utc)
    recency_days = float((now_utc - last_ts.to_pydatetime()).total_seconds() / 86400.0)

    return Features(
        ret_7=ret_7,
        ret_30=ret_30,
        risk=risk,
        drawdown_30=drawdown_30,
        value_z=value_z,
        liq_7=liq_7,
        recency_days=recency_days,
    )
