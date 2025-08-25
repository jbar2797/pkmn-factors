from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..config import Settings
from ..eval.backtest import run as run_backtest
from ..ingest.csv_to_trades import ingest_csv
from .schemas import BacktestRequest, HealthResponse, IngestCSVRequest

# Load env-backed config (expects DATABASE_URL)
_settings = Settings()

_engine: Optional[AsyncEngine] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    _engine = create_async_engine(_settings.DATABASE_URL, future=True)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if _engine is not None:
            await _engine.dispose()


app = FastAPI(title="pkmn-factors API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    assert _engine is not None
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return HealthResponse(status="ok", db="up")
    except Exception as e:  # pragma: no cover - surface raw for now
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/csv")
async def api_ingest_csv(req: IngestCSVRequest) -> dict:
    try:
        inserted = await ingest_csv(
            Path(req.path),
            source=req.source,
            currency=req.currency,
            card_key_override=req.card_key_override,
        )
        return {"inserted": inserted}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/backtest")
async def api_backtest(req: BacktestRequest) -> dict:
    try:
        await run_backtest(
            req.card_key,
            req.model_version,
            req.horizon_days,
            req.persist,
            req.note,
        )
        return {"ok": True}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(e))
