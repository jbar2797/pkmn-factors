from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy Declarative Base."""

    pass


# ----------------------------
# Core domain tables
# ----------------------------


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    card_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    set_code: Mapped[str] = mapped_column(String(32))
    rarity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    release_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Card(card_key={self.card_key!r}, name={self.name!r})"


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(32))
    card_key: Mapped[str] = mapped_column(String(128), index=True)
    grade: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    listing_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    price: Mapped[float] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(8))
    link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # de-dup guard (works well for scraped/tracked sources)
        UniqueConstraint("timestamp", "price", "source", "link", name="uq_trade_dedup"),
        Index("ix_trades_card_time", "card_key", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Trade(card_key={self.card_key!r}, price={self.price!r}, ts={self.timestamp!r})"


# ----------------------------
# Model signals (outputs)
# ----------------------------


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # when the signal was produced (server default = now())
    asof_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # which card the signal pertains to
    card_key: Mapped[str] = mapped_column(String(128), index=True)

    # model outputs
    horizon_days: Mapped[int] = mapped_column(Integer, default=90)
    action: Mapped[str] = mapped_column(String(8))  # BUY | HOLD | SELL
    conviction: Mapped[float] = mapped_column(Float)  # 0..1 confidence
    expected_return: Mapped[float] = mapped_column(Float)  # e.g., 0.05 = +5%
    risk: Mapped[float] = mapped_column(Float)  # annualized-ish stdev proxy
    utility: Mapped[float] = mapped_column(Float)  # e.g., Sharpe-like

    # versioning + optional features snapshot
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    __table_args__ = (Index("ix_signals_card_asof", "card_key", "asof_ts"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Signal(card_key={self.card_key!r}, action={self.action!r}, asof={self.asof_ts!r})"
