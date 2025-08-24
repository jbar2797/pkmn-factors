from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import pandas as pd


WEIGHTS = {
    "rarity_type": 10,
    "pop_level": 8,
    "pop_velocity": 12,
    "psa10_to_9_ratio": 8,
    "gem_difficulty": 8,
    "momentum_short": 12,
    "euphoria_penalty": 6,
    "drawdown_short": 8,
    "drawdown_long": 6,
    "relative_value": 10,
    "liquidity": 6,
    "popularity": 4,
    "set_age": 2,
}


def _tricube(days_ago: float, bandwidth: float = 7.0) -> float:
    if days_ago < 0:
        return 0.0
    u = min(days_ago / bandwidth, 4.0)
    if u >= 1.0:
        return max(0.0, (1 - u) ** 3)
    return (1 - u**3) ** 3


def robust_now_price(
    trades: pd.DataFrame, asof: pd.Timestamp | None = None
) -> float | None:
    """
    Recency-weighted robust 'now' price using
    - tricube kernel on days_ago
    - 2.5–97.5% trim
    - median-of-means across 5 buckets (if enough data)
    """
    if trades is None or trades.empty:
        return None

    t = trades.copy()

    # Ensure 'asof' is a concrete Timestamp
    if asof is None:
        asof = t["timestamp"].max()
    asof_ts: pd.Timestamp = pd.Timestamp(asof)

    # Ensure datetime dtype for arithmetic
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")

    t = t.sort_values("timestamp")
    # Use Series - Timestamp (supported by pandas-stubs), then negate to get "ago"
    t["days_ago"] = (t["timestamp"] - asof_ts).dt.days * -1
    t = t[t["days_ago"] <= 90]
    if t.empty:
        return None

    t["w"] = t["days_ago"].apply(lambda d: _tricube(float(d)))
    if "listing_type" in t:
        t.loc[t["listing_type"].str.lower() == "auction", "w"] *= 1.15

    lo, hi = np.percentile(t["price"], [2.5, 97.5])
    t = t[(t["price"] >= lo) & (t["price"] <= hi)]
    if t.empty:
        return None

    sorted_t = t.sort_values("days_ago")
    buckets = np.array_split(sorted_t, 5) if len(sorted_t) >= 10 else [sorted_t]
    moms: list[float] = []
    for b in buckets:
        if b["w"].sum() > 0:
            moms.append(float(np.average(b["price"], weights=b["w"])))
        else:
            moms.append(float(b["price"].mean()))
    return float(np.median(moms))


def rolling_stats(trades: pd.DataFrame, asof: pd.Timestamp | None = None):
    """
    Return (avg7, high30, high52) from trades.
    """
    if trades is None or trades.empty:
        return None, None, None

    if asof is None:
        asof = trades["timestamp"].max()

    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")

    d7 = t[t["timestamp"] >= pd.Timestamp(asof) - pd.Timedelta(days=7)]
    d30 = t[t["timestamp"] >= pd.Timestamp(asof) - pd.Timedelta(days=30)]
    d52 = t[t["timestamp"] >= pd.Timestamp(asof) - pd.Timedelta(days=365)]

    avg7 = float(d7["price"].mean()) if not d7.empty else None
    high30 = float(d30["price"].max()) if not d30.empty else None
    high52 = float(d52["price"].max()) if not d52.empty else None
    return avg7, high30, high52


def rarity_type_score(flag: Optional[str]) -> float:
    tiers = {
        "alt_art": 1.0,
        "promo": 0.9,
        "secret": 0.85,
        "full_art": 0.75,
        "standard": 0.45,
    }
    return tiers.get((flag or "").lower(), 0.5)


def pop_level_score(psa10: int | None) -> float:
    if not psa10:
        return 0.5
    if psa10 <= 250:
        return 1.0
    if psa10 <= 1000:
        return 0.8
    if psa10 <= 3000:
        return 0.65
    if psa10 <= 6000:
        return 0.5
    if psa10 <= 10000:
        return 0.4
    return 0.3


def pop_velocity_score(delta_30d: int | None, psa10: int | None) -> float:
    if not psa10:
        return 0.5
    pct = (delta_30d or 0) / psa10
    if pct <= 0.005:
        return 1.0
    if pct <= 0.01:
        return 0.9
    if pct <= 0.02:
        return 0.75
    if pct <= 0.05:
        return 0.55
    if pct <= 0.10:
        return 0.4
    return 0.25


def ratio_score(psa10: int | None, psa9: int | None) -> float:
    if not psa10 or not psa9:
        return 0.5
    r = psa10 / psa9
    if r <= 0.1:
        return 0.45
    if r <= 0.2:
        return 0.6
    if r <= 0.35:
        return 0.75
    if r <= 0.5:
        return 0.85
    return 0.95


def gem_difficulty_score(gem_pct: float | None) -> float:
    if gem_pct is None:
        return 0.5
    if gem_pct <= 10:
        return 1.0
    if gem_pct <= 20:
        return 0.9
    if gem_pct <= 30:
        return 0.8
    if gem_pct <= 40:
        return 0.65
    if gem_pct <= 50:
        return 0.5
    return 0.35


def momentum_short_score(now: float | None, avg7: float | None) -> float:
    if not now or not avg7:
        return 0.5
    m = (now - avg7) / avg7
    if -0.10 <= m <= 0.08:
        return 1.0 - abs(m) * 0.8
    if m < -0.10:
        return 0.65
    if m <= 0.15:
        return 0.55
    return 0.35


def euphoria_penalty(
    now: float | None, high30: float | None, sales_14d: int | None
) -> float:
    if not now or not high30 or high30 <= 0:
        return 0.5
    pct = now / high30
    thin = (sales_14d or 0) < 10
    if pct >= 0.98 and thin:
        return 0.2
    if pct >= 0.98:
        return 0.4
    if pct >= 0.95 and thin:
        return 0.45
    return 0.7


def drawdown_score(now: float | None, ref: float | None) -> float:
    if not now or not ref or ref <= 0:
        return 0.5
    dd = 1 - (now / ref)
    if 0.10 <= dd <= 0.40:
        return 0.95
    if dd < 0.10 and dd >= 0.05:
        return 0.75
    if dd <= 0.60 and dd > 0.40:
        return 0.6
    if dd < 0.05:
        return 0.4
    return 0.5


def relative_value_score(now: float | None, cohort: float | None) -> float:
    if not now or not cohort or cohort <= 0:
        return 0.5
    rel = now / cohort
    if 0.70 <= rel <= 0.95:
        return 0.95
    if rel <= 1.10:
        return 0.8
    if rel < 0.70:
        return 0.6
    if rel <= 1.50:
        return 0.5
    return 0.35


def liquidity_score(sales_30d: int | None) -> float:
    if sales_30d is None:
        return 0.4
    if sales_30d >= 60:
        return 1.0
    if sales_30d >= 30:
        return 0.9
    if sales_30d >= 15:
        return 0.75
    if sales_30d >= 6:
        return 0.6
    if sales_30d >= 1:
        return 0.45
    return 0.35


def popularity_score(rank: int | None, trends: float | None) -> float:
    rank_term = (
        1.0
        if not rank or rank <= 5
        else 0.8 if rank <= 15 else 0.6 if rank <= 30 else 0.45
    )
    trends_term = (
        0.35 if trends is None else 0.35 + 0.65 * (max(0, min(100, trends)) / 100.0)
    )
    return 0.4 * rank_term + 0.6 * trends_term


def set_age_score(months: int | None) -> float:
    if months is None:
        return 0.5
    if 12 <= months <= 36:
        return 1.0
    if 6 <= months < 12:
        return 0.75
    if 36 < months <= 60:
        return 0.8
    if months < 6:
        return 0.45
    return 0.7


def compute_score_rt(row: dict, trades: pd.DataFrame | None = None) -> dict:
    """
    Main scoring function. If 'trades' is provided, price anchors and sales counts
    are derived from the tape; otherwise values in 'row' are used.
    """
    # Derive set age (months) if not provided
    age_m = row.get("set_age_months")
    if not age_m and row.get("set_release_date") and row.get("asof_date"):
        sr = pd.to_datetime(cast(Any, row["set_release_date"]))
        asof_dt = pd.to_datetime(cast(Any, row["asof_date"]))
        age_m = max(0, (asof_dt.year - sr.year) * 12 + (asof_dt.month - sr.month))

    # Anchors (may be overridden by trades)
    now = row.get("price_now")
    row_avg7 = row.get("price_avg_7d")
    high30 = row.get("price_high_30d")
    high52 = row.get("price_high_52w")
    sales_14d = row.get("sales_14d")
    row_sales_30d = row.get("sales_30d")

    if trades is not None and not trades.empty:
        asof = (
            pd.to_datetime(cast(Any, row.get("asof_date")))
            if row.get("asof_date")
            else trades["timestamp"].max()
        )
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        if now is None:
            now = robust_now_price(t, pd.Timestamp(asof))
        row_avg7, high30, high52 = rolling_stats(t, pd.Timestamp(asof))
        sales_14d = int(
            (t["timestamp"] >= pd.Timestamp(asof) - pd.Timedelta(days=14)).sum()
        )
        row_sales_30d = int(
            (t["timestamp"] >= pd.Timestamp(asof) - pd.Timedelta(days=30)).sum()
        )

    subs = {
        "rarity_type": rarity_type_score(cast(Optional[str], row.get("rarity_flag"))),
        "pop_level": pop_level_score(cast(Optional[int], row.get("psa10_pop")) or None),
        "pop_velocity": pop_velocity_score(
            cast(Optional[int], row.get("psa10_pop_30d_change")) or None,
            cast(Optional[int], row.get("psa10_pop")) or None,
        ),
        "psa10_to_9_ratio": ratio_score(
            cast(Optional[int], row.get("psa10_pop")) or None,
            cast(Optional[int], row.get("psa9_pop")) or None,
        ),
        "gem_difficulty": gem_difficulty_score(
            cast(Optional[float], row.get("gem_rate_pct"))
        ),
        "momentum_short": momentum_short_score(
            cast(Optional[float], now), cast(Optional[float], row_avg7)
        ),
        "euphoria_penalty": euphoria_penalty(
            cast(Optional[float], now),
            cast(Optional[float], high30),
            cast(Optional[int], sales_14d),
        ),
        "drawdown_short": drawdown_score(
            cast(Optional[float], now), cast(Optional[float], high30)
        ),
        "drawdown_long": drawdown_score(
            cast(Optional[float], now), cast(Optional[float], high52)
        ),
        "relative_value": relative_value_score(
            cast(Optional[float], now), cast(Optional[float], row.get("cohort_median"))
        ),
        "liquidity": liquidity_score(cast(Optional[int], row_sales_30d)),
        "popularity": popularity_score(
            cast(Optional[int], row.get("species_popularity_rank")),
            cast(Optional[float], row.get("google_trends_90d")),
        ),
        "set_age": set_age_score(cast(Optional[int], age_m)),
    }

    total = sum(WEIGHTS[k] * subs[k] for k in WEIGHTS)
    score = float(total)  # weights sum to 100 → already in 0–100 scale
    if score >= 80:
        decision = "BUY (Strong)"
    elif score >= 65:
        decision = "BUY (Selective)"
    elif score >= 50:
        decision = "HOLD / WATCH"
    else:
        decision = "PASS"

    return {
        "score": round(score, 1),
        "decision": decision,
        "now_price_used": None if now is None else float(now),
        "subscores": {k: round(v, 3) for k, v in subs.items()},
    }
