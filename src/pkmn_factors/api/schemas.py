from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class IngestCSVRequest(BaseModel):
    path: str = Field(..., description="Absolute or repo-relative path to CSV")
    source: str = "api"
    currency: str = "USD"
    # If provided, every row will be forced to this card key
    card_key_override: Optional[str] = None


class BacktestRequest(BaseModel):
    card_key: str
    model_version: str
    horizon_days: int = 90
    persist: bool = False
    note: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    db: str
