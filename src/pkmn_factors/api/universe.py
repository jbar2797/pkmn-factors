# src/pkmn_factors/api/universe.py
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Sequence

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pkmn_factors.config import settings

# --- DB session factory (async) ----------------------------------------------
_engine = create_async_engine(
    settings.database_url, future=True
)  # expects env DATABASE_URL
_SessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

router = APIRouter(tags=["universe"])


class TopMetric(BaseModel):
    card_key: str
    model_version: str
    horizon_days: int
    cum_return: float
    sharpe: Optional[float] = None
    asof_ts: datetime


async def _fetch_top(
    limit: int,
    model_version: Optional[str],
    horizon_days: Optional[int],
) -> List[TopMetric]:
    """Fetch top rows from metrics ordered by Sharpe (NULLS LAST), then cum_return."""
    q = text(
        """
        SELECT card_key, model_version, horizon_days,
               COALESCE(cum_return, 0.0) AS cum_return,
               sharpe, asof_ts
        FROM metrics
        WHERE (:model_version IS NULL OR model_version = :model_version)
          AND (:horizon_days IS NULL OR horizon_days = :horizon_days)
        ORDER BY sharpe DESC NULLS LAST, cum_return DESC NULLS LAST, asof_ts DESC
        LIMIT :limit
        """
    )

    params = {
        "model_version": model_version,
        "horizon_days": horizon_days,
        "limit": int(limit),
    }

    async with _SessionLocal() as session:
        rows: Sequence[Any] = (await session.execute(q, params)).mappings().all()  # type: ignore[assignment]
    out: List[TopMetric] = []
    for r in rows:
        # SQL numeric may come back as Decimal; coerce to float explicitly
        out.append(
            TopMetric(
                card_key=r["card_key"],
                model_version=r["model_version"],
                horizon_days=int(r["horizon_days"]),
                cum_return=(
                    float(r["cum_return"]) if r["cum_return"] is not None else 0.0
                ),
                sharpe=float(r["sharpe"]) if r["sharpe"] is not None else None,
                asof_ts=r["asof_ts"],
            )
        )
    return out


@router.get("/universe/top", response_model=List[TopMetric])
async def universe_top(
    limit: int = 10,
    model_version: Optional[str] = None,
    horizon_days: Optional[int] = None,
) -> List[TopMetric]:
    """JSON: top metrics snapshot."""
    return await _fetch_top(
        limit=limit, model_version=model_version, horizon_days=horizon_days
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(limit: int = 10) -> HTMLResponse:
    """Ultra-light HTML view (no templates) to eyeball the universe."""
    rows = await _fetch_top(limit=limit, model_version=None, horizon_days=None)

    def fmt(x: Optional[float]) -> str:
        if x is None:
            return "—"
        return f"{x:.4f}"

    # build a tiny HTML table
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>PKMN Universe</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px}table{border-collapse:collapse}th,td{padding:8px 12px;border:1px solid #ddd}th{background:#f6f6f6}code{background:#f1f1f1;padding:2px 6px;border-radius:4px}</style>",
        "</head><body>",
        "<h2>Universe snapshot</h2>",
        "<p>Top rows by <code>sharpe</code>, then <code>cum_return</code>. Limit="
        f"<code>{limit}</code></p>",
        "<table><thead><tr>",
        "<th>Card</th><th>Model</th><th>Horizon</th><th>Sharpe</th><th>Cum Return</th><th>As Of</th>",
        "</tr></thead><tbody>",
    ]
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{r.card_key}</td>"
            f"<td>{r.model_version}</td>"
            f"<td style='text-align:right'>{r.horizon_days}</td>"
            f"<td style='text-align:right'>{fmt(r.sharpe)}</td>"
            f"<td style='text-align:right'>{fmt(r.cum_return)}</td>"
            f"<td>{r.asof_ts.isoformat()}</td>"
            "</tr>"
        )
    body += ["</tbody></table>", "</body></html>"]
    return HTMLResponse("\n".join(body))
