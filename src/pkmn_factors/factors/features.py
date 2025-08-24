from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd


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


# ---------- helpers ----------


def _to_float_series(s: pd.Series) -> pd.Series:
    """Ensure a numeric (float) series even if input has Decimals/strings."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_pct_change(s: pd.Series, periods: int) -> float:
    """(last / value N periods ago) - 1, with safety checks."""
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
    """Std dev of daily returns over a rolling window."""
    try:
        r = s.pct_change().dropna().tail(window)
        if r.empty:
            return float("nan")
        v = r.std(ddof=1)
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def _max_drawdown(s: pd.Series, window: int) -> float:
    """Max drawdown magnitude over a window (positive number)."""
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
        return float(abs(mdd))
    except Exception:
        return float("nan")


def _z_score_latest(s: pd.Series, window: int) -> float:
    """Z-score of the latest value vs a rolling window."""
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


# ---------- main feature builder ----------


def build_features(df: pd.DataFrame) -> Features:
    """
    Build the feature vector from raw trades/prices.

    Required columns in `df`:
      - 'timestamp' (datetime-like; tz-naive or tz-aware OK)
      - 'price' (numeric)
    """
    if df.empty:
        return Features(*(float("nan"),) * 7)

    # normalize types
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["price"] = _to_float_series(d["price"])

    # daily last observed price series
    daily = d.resample("1D", on="timestamp")["price"].last().dropna()
    if daily.shape[0] == 0:
        return Features(*(float("nan"),) * 7)

    # price/return/risks
    ret_7 = _safe_pct_change(daily, 7)
    ret_30 = _safe_pct_change(daily, 30)
    risk = _rolling_vol(daily, 30)
    drawdown_30 = _max_drawdown(daily, 30)
    value_z = _z_score_latest(daily, 30)

    # liquidity: number of raw rows in last 7 days (no .last('7D') warning)
    try:
        last_ts = pd.to_datetime(d["timestamp"].max(), utc=True)
        start = last_ts - pd.Timedelta(days=7)
        liq_7 = float(d[d["timestamp"] >= start].shape[0]) if not d.empty else 0.0
    except Exception:
        liq_7 = 0.0

    # recency: days from most recent trade to now (UTC)
    try:
        last_ts = pd.to_datetime(d["timestamp"].max(), utc=True)
        now_utc = datetime.now(timezone.utc)
        recency_days = float(
            (now_utc - last_ts.to_pydatetime()).total_seconds() / 86400.0
        )
    except Exception:
        recency_days = float("nan")

    return Features(
        ret_7=ret_7,
        ret_30=ret_30,
        risk=risk,
        drawdown_30=drawdown_30,
        value_z=value_z,
        liq_7=liq_7,
        recency_days=recency_days,
    )
