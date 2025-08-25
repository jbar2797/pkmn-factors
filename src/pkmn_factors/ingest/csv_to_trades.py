from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from pkmn_factors.config import settings

# Executemany payload item (named binds)
RowMap = Dict[str, Any]


def _to_utc(s: pd.Series) -> pd.Series:
    """Coerce datetimes to tz-aware UTC."""
    return pd.to_datetime(s, utc=True)


def _load_csv_rows(
    csv_path: Path,
    source: str,
    override_card_key: Optional[str] = None,
    currency_default: str = "USD",
) -> List[RowMap]:
    """
    Read CSV and build rows for INSERT.
    Expected CSV columns minimum: timestamp, price, card_key
    Optional CSV column: currency  (if missing -> currency_default)
    """
    df = pd.read_csv(csv_path)

    required = {"timestamp", "price", "card_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Normalize
    df["timestamp"] = _to_utc(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["card_key"] = df["card_key"].astype(str)

    # If the CSV contains currency, use it; otherwise fill default
    if "currency" in df.columns:
        df["currency"] = df["currency"].fillna(currency_default).astype(str)
    else:
        df["currency"] = currency_default

    df = df[["timestamp", "price", "card_key", "currency"]].dropna()

    rows: List[RowMap] = []
    for ts, price, ck, ccy in df.itertuples(index=False, name=None):  # type: ignore[misc]
        # Ensure python-native types
        if hasattr(ts, "to_pydatetime"):
            ts_dt: datetime = ts.to_pydatetime().astimezone(timezone.utc)
        else:
            ts_dt = pd.to_datetime(ts, utc=True).to_pydatetime()

        rows.append(
            {
                "timestamp": ts_dt,
                "source": source,
                "card_key": str(override_card_key) if override_card_key else str(ck),
                "price": float(price),
                "currency": str(ccy),
            }
        )

    return rows


async def _bulk_insert(engine: AsyncEngine, rows: List[RowMap]) -> int:
    """
    Insert rows into trades with explicit column list including NOT NULL fields.
    Uses named binds -> pass a list of dicts.
    """
    if not rows:
        return 0

    q = text(
        """
        INSERT INTO trades (timestamp, source, card_key, price, currency)
        VALUES (:timestamp, :source, :card_key, :price, :currency)
        """
    )

    async with engine.begin() as conn:
        await conn.execute(q, rows)

    return len(rows)


async def ingest_csv(
    csv_path: Path,
    *,
    source: str = "demo_csv",
    override_card_key: Optional[str] = None,
    currency_default: str = "USD",
) -> int:
    """
    Public entrypoint: read CSV and insert into trades.
    """
    engine = create_async_engine(settings.database_url, future=True)  # type: ignore[attr-defined]
    rows = _load_csv_rows(
        csv_path,
        source,
        override_card_key=override_card_key,
        currency_default=currency_default,
    )
    return await _bulk_insert(engine, rows)


# Optional CLI:
#   uv run python -m pkmn_factors.ingest.csv_to_trades \
#       --csv data/trades_demo.csv --source demo_csv --card-key mew-ex-053-svp-2023 --currency USD
if __name__ == "__main__":
    import argparse
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/trades_demo.csv"))
    ap.add_argument("--source", type=str, default="demo_csv")
    ap.add_argument("--card-key", type=str, default=None)
    ap.add_argument("--currency", type=str, default="USD")
    args = ap.parse_args()

    asyncio.run(
        ingest_csv(
            args.csv,
            source=args.source,
            override_card_key=args.card_key,
            currency_default=args.currency,
        )
    )
