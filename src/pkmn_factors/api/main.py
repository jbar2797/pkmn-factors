# src/pkmn_factors/api/main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pkmn_factors.api.universe import router as universe_router
from pkmn_factors.config import settings  # type: ignore[attr-defined]
from pkmn_factors.eval.backtest import run as run_backtest
from pkmn_factors.ingest.csv_to_trades import ingest_csv
from pkmn_factors.universe import load_universe

app = FastAPI(title="pkmn-factors API")

# ---- DB session factory ----
engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@app.get("/health")
async def health() -> dict[str, str]:
    # simple connectivity check
    async with engine.begin():
        pass
    return {"status": "ok"}


# ---- ingest CSV ----
class IngestCSVRequest(BaseModel):
    path: str
    source: str = "api"
    currency: str = "USD"
    card_key_override: Optional[str] = None


@app.post("/ingest/csv")
async def ingest_csv_endpoint(req: IngestCSVRequest) -> dict[str, int]:
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(400, f"File not found: {p}")
    inserted = await ingest_csv(
        p,
        source=req.source,
        currency=req.currency,
        card_key_override=req.card_key_override,
    )
    return {"inserted": inserted}


# ---- backtest ----
class BacktestRequest(BaseModel):
    card_key: str
    model_version: str
    horizon_days: int = 90
    persist: bool = False
    note: Optional[str] = None


@app.post("/backtest")
async def backtest_endpoint(req: BacktestRequest) -> dict[str, str]:
    await run_backtest(
        req.card_key, req.model_version, req.horizon_days, req.persist, req.note
    )
    return {"status": "ok"}


# ---- latest metrics ----
@app.get("/metrics/latest")
async def metrics_latest(limit: int = 25) -> list[dict]:
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    text(
                        """
                        SELECT asof_ts, card_key, model_version, horizon_days, cum_return, sharpe
                        FROM metrics
                        ORDER BY asof_ts DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
            )
            .mappings()
            .all()
        )
        return [dict(r) for r in rows]


# ---- universe (file-backed) ----
@app.get("/universe")
async def universe(path: str = "data/universe_demo.csv") -> list[str]:
    return sorted(load_universe(Path(path)))


# ---- static UI (/ui) ----
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

# ---- include extra API routes (/universe/top, /dashboard, etc.) ----
app.include_router(universe_router)
