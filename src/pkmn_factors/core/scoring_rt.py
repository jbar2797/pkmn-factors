from __future__ import annotations
import numpy as np
import pandas as pd

WEIGHTS = {
    "rarity_type": 10, "pop_level": 8, "pop_velocity": 12, "psa10_to_9_ratio": 8,
    "gem_difficulty": 8, "momentum_short": 12, "euphoria_penalty": 6,
    "drawdown_short": 8, "drawdown_long": 6, "relative_value": 10,
    "liquidity": 6, "popularity": 4, "set_age": 2,
}

def _tricube(days_ago: float, bandwidth: float = 7.0) -> float:
    if days_ago < 0: return 0.0
    u = min(days_ago / bandwidth, 4.0)
    return max(0.0, (1 - u) ** 3) if u >= 1.0 else (1 - u**3) ** 3

def robust_now_price(trades: pd.DataFrame, asof: pd.Timestamp | None = None) -> float | None:
    if trades is None or trades.empty: return None
    t = trades.copy()
    if asof is None: asof = t["timestamp"].max()
    t = t.sort_values("timestamp")
    t["days_ago"] = (asof - t["timestamp"]).dt.days
    t = t[t["days_ago"] <= 90]
    if t.empty: return None
    t["w"] = t["days_ago"].apply(lambda d: _tricube(float(d)))
    if "listing_type" in t:
        t.loc[t["listing_type"].str.lower() == "auction", "w"] *= 1.15
    lo, hi = np.percentile(t["price"], [2.5, 97.5])
    t = t[(t["price"] >= lo) & (t["price"] <= hi)]
    buckets = np.array_split(t.sort_values("days_ago"), 5) if len(t) >= 10 else [t]
    moms = [np.average(b["price"], weights=b["w"]) if b["w"].sum() > 0 else b["price"].mean() for b in buckets]
    return float(np.median(moms))

def rolling_stats(trades: pd.DataFrame, asof: pd.Timestamp | None = None):
    if trades is None or trades.empty: return None, None, None
    if asof is None: asof = trades["timestamp"].max()
    d7  = trades[trades["timestamp"] >= asof - pd.Timedelta(days=7)]
    d30 = trades[trades["timestamp"] >= asof - pd.Timedelta(days=30)]
    d52 = trades[trades["timestamp"] >= asof - pd.Timedelta(days=365)]
    avg7 = float(d7["price"].mean()) if not d7.empty else None
    high30 = float(d30["price"].max()) if not d30.empty else None
    high52 = float(d52["price"].max()) if not d52.empty else None
    return avg7, high30, high52

def rarity_type_score(flag:str)->float:
    tiers={"alt_art":1.0,"promo":0.9,"secret":0.85,"full_art":0.75,"standard":0.45}
    return tiers.get((flag or "").lower(),0.5)

def pop_level_score(psa10:int|None)->float:
    if not psa10: return 0.5
    return 1.0 if psa10<=250 else 0.8 if psa10<=1000 else 0.65 if psa10<=3000 else 0.5 if psa10<=6000 else 0.4 if psa10<=10000 else 0.3

def pop_velocity_score(delta_30d:int|None, psa10:int|None)->float:
    if not psa10: return 0.5
    pct=(delta_30d or 0)/psa10
    return 1.0 if pct<=0.005 else 0.9 if pct<=0.01 else 0.75 if pct<=0.02 else 0.55 if pct<=0.05 else 0.4 if pct<=0.10 else 0.25

def ratio_score(psa10:int|None, psa9:int|None)->float:
    if not psa10 or not psa9: return 0.5
    r=psa10/psa9
    return 0.45 if r<=0.1 else 0.6 if r<=0.2 else 0.75 if r<=0.35 else 0.85 if r<=0.5 else 0.95

def gem_difficulty_score(gem_pct:float|None)->float:
    if gem_pct is None: return 0.5
    return 1.0 if gem_pct<=10 else 0.9 if gem_pct<=20 else 0.8 if gem_pct<=30 else 0.65 if gem_pct<=40 else 0.5 if gem_pct<=50 else 0.35

def momentum_short_score(now:float|None, avg7:float|None)->float:
    if not now or not avg7: return 0.5
    m=(now-avg7)/avg7
    return (1.0-abs(m)*0.8) if -0.10<=m<=0.08 else 0.65 if m<-0.10 else 0.55 if m<=0.15 else 0.35

def euphoria_penalty(now:float|None, high30:float|None, sales_14d:int|None)->float:
    if not now or not high30 or high30<=0: return 0.5
    pct=now/high30; thin=(sales_14d or 0)<10
    if pct>=0.98 and thin: return 0.2
    if pct>=0.98: return 0.4
    if pct>=0.95 and thin: return 0.45
    return 0.7

def drawdown_score(now:float|None, ref:float|None)->float:
    if not now or not ref or ref<=0: return 0.5
    dd=1-(now/ref)
    return 0.95 if 0.10<=dd<=0.40 else 0.75 if dd<0.10 and dd>=0.05 else 0.6 if dd<=0.60 and dd>0.40 else 0.4 if dd<0.05 else 0.5

def relative_value_score(now:float|None, cohort:float|None)->float:
    if not now or not cohort or cohort<=0: return 0.5
    rel=now/cohort
    return 0.95 if 0.70<=rel<=0.95 else 0.8 if rel<=1.10 else 0.6 if rel<0.70 else 0.5 if rel<=1.50 else 0.35

def liquidity_score(sales_30d:int|None)->float:
    if sales_30d is None: return 0.4
    return 1.0 if sales_30d>=60 else 0.9 if sales_30d>=30 else 0.75 if sales_30d>=15 else 0.6 if sales_30d>=6 else 0.45 if sales_30d>=1 else 0.35

def popularity_score(rank:int|None, trends:float|None)->float:
    rank_term=1.0 if not rank or rank<=5 else 0.8 if rank<=15 else 0.6 if rank<=30 else 0.45
    trends_term=0.35 if trends is None else 0.35+0.65*(max(0,min(100,trends))/100.0)
    return 0.4*rank_term + 0.6*trends_term

def set_age_score(months:int|None)->float:
    if months is None: return 0.5
    return 1.0 if 12<=months<=36 else 0.75 if 6<=months<12 else 0.8 if 36<months<=60 else 0.45 if months<6 else 0.7

def compute_score_rt(row:dict, trades:pd.DataFrame|None=None)->dict:
    age_m=row.get("set_age_months")
    if not age_m and row.get("set_release_date") and row.get("asof_date"):
        sr=pd.to_datetime(row["set_release_date"]); asof=pd.to_datetime(row["asof_date"])
        age_m=max(0,(asof.year-sr.year)*12+(asof.month-sr.month))

    now,row_avg7,high30,high52 = row.get("price_now"),row.get("price_avg_7d"),row.get("price_high_30d"),row.get("price_high_52w")
    sales_14d,row_sales_30d=row.get("sales_14d"),row.get("sales_30d")
    if trades is not None and not trades.empty:
        asof=pd.to_datetime(row.get("asof_date")) if row.get("asof_date") else trades["timestamp"].max()
        trades=trades.copy(); trades["timestamp"]=pd.to_datetime(trades["timestamp"])
        now = now or robust_now_price(trades, asof)
        row_avg7,high30,high52 = rolling_stats(trades, asof)
        sales_14d = int((trades["timestamp"] >= (asof - pd.Timedelta(days=14))).sum())
        row_sales_30d = int((trades["timestamp"] >= (asof - pd.Timedelta(days=30))).sum())

    subs={
        "rarity_type": rarity_type_score(row.get("rarity_flag")),
        "pop_level": pop_level_score(row.get("psa10_pop")),
        "pop_velocity": pop_velocity_score(row.get("psa10_pop_30d_change"), row.get("psa10_pop")),
        "psa10_to_9_ratio": ratio_score(row.get("psa10_pop"), row.get("psa9_pop")),
        "gem_difficulty": gem_difficulty_score(row.get("gem_rate_pct")),
        "momentum_short": momentum_short_score(now, row_avg7),
        "euphoria_penalty": euphoria_penalty(now, high30, sales_14d),
        "drawdown_short": drawdown_score(now, high30),
        "drawdown_long": drawdown_score(now, high52),
        "relative_value": relative_value_score(now, row.get("cohort_median")),
        "liquidity": liquidity_score(row_sales_30d),
        "popularity": popularity_score(row.get("species_popularity_rank"), row.get("google_trends_90d")),
        "set_age": set_age_score(age_m),
    }
    total=sum(WEIGHTS[k]*subs[k] for k in WEIGHTS)
    score=(total/100.0)*100.0
    decision="BUY (Strong)" if score>=80 else "BUY (Selective)" if score>=65 else "HOLD / WATCH" if score>=50 else "PASS"
    return {
        "score": round(score,1), "decision": decision, "now_price_used": None if now is None else float(now),
        "subscores": {k: round(v,3) for k,v in subs.items()},
    }
