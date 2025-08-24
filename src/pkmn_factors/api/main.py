from __future__ import annotations

from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pandas as pd

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from pkmn_factors.api.demo import router as demo_router
from pkmn_factors.core.scoring_rt import compute_score_rt

# DB plumbing for health check
from pkmn_factors.db.base import get_session
from pkmn_factors.db.models import Trade


app = FastAPI(title="PKMN Factors (RT)")

# existing demo routes
app.include_router(demo_router)


class ScorePayload(BaseModel):
    row: dict
    trades_csv_path: str | None = None


@app.post("/score")
def score(payload: ScorePayload):
    trades = (
        pd.read_csv(payload.trades_csv_path, parse_dates=["timestamp"])
        if payload.trades_csv_path
        else None
    )
    return compute_score_rt(payload.row, trades)


# -------- Health / DB endpoint --------
@app.get("/health/db")
async def health_db(session: AsyncSession = Depends(get_session)):
    # simple ping
    await session.execute(text("SELECT 1"))
    # count rows in trades hypertable
    result = await session.execute(select(func.count()).select_from(Trade))
    count = result.scalar_one()
    return {"ok": True, "trades": int(count)}
