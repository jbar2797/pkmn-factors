from fastapi import APIRouter, Depends
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pkmn_factors.db.base import get_session
from pkmn_factors.db.models import Trade

router = APIRouter()


@router.get("/health/db")
async def health_db(session: AsyncSession = Depends(get_session)):
    await session.execute(text("SELECT 1"))
    result = await session.execute(select(func.count()).select_from(Trade))
    count = result.scalar_one()
    return {"ok": True, "trades": int(count)}
