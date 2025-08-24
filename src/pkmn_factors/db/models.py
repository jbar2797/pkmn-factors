from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Numeric,
    DateTime,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from pkmn_factors.db.base import Base


class Trade(Base):
    """
    Ultra-modern Pokémon TCG trade (sold listing) record.

    Notes (TimescaleDB):
      - Hypertable will partition on `timestamp`
      - Any UNIQUE / PRIMARY KEY on a hypertable must include the partition key.
        Therefore we use a composite PK: (id, timestamp).
    """

    __tablename__ = "trades"

    # Composite primary key: id + timestamp
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # MUST be part of the PK because it's the partitioning (time) column
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, index=True, nullable=False
    )

    source: Mapped[str] = mapped_column(String(32), nullable=False)  # ebay, pwcc, etc.
    card_key: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # e.g., "SVP-053 Mew ex PSA10"

    grade: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # PSA10, PSA9
    listing_type: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # auction, BIN

    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # de-dup within a minute/price/src/link combo
        UniqueConstraint("timestamp", "price", "source", "link", name="uq_trade_dedup"),
        Index("ix_trades_card_time", "card_key", "timestamp"),
    )
