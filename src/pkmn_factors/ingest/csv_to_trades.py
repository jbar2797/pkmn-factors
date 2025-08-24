from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from pkmn_factors.db.base import get_engine


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV and normalize column names/types."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Accept either 'ts' or 'timestamp' in the CSV, normalize to 'timestamp'
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})

    required = {"timestamp", "price", "card_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["card_key"] = df["card_key"].astype(str)

    # Drop rows with any nulls in required fields
    df = df.dropna(subset=["timestamp", "price", "card_key"])

    return df[["timestamp", "price", "card_key"]]


async def _bulk_insert(engine: AsyncEngine, rows: Iterable[dict]) -> int:
    """
    Insert rows into trades. Assumes table schema:
      trades(timestamp timestamptz, price numeric, card_key text)
    """
    q = text(
        """
        INSERT INTO trades (timestamp, price, card_key)
        VALUES (:timestamp, :price, :card_key)
        """
    )

    count = 0
    async with engine.begin() as conn:
        # Executemany with a list[dict]
        rp = await conn.execute(q, list(rows))
        # Some drivers don't report rowcount reliably in executemany; be tolerant.
        count = getattr(rp, "rowcount", 0) or 0
    return int(count)


async def run(csv_path_str: str) -> None:
    csv_path = Path(csv_path_str)
    df = _load_csv(csv_path)

    engine = get_engine()  # <- get a real AsyncEngine (not a callable)
    payload = (
        {
            "timestamp": pd.to_datetime(ts, utc=True).to_pydatetime(),
            "price": float(price),
            "card_key": str(card),
        }
        for ts, price, card in df.itertuples(index=False, name=None)
    )

    inserted = await _bulk_insert(engine, payload)
    print(f"Inserted ~{inserted} rows from {csv_path.name}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Load a CSV of trades into the DB.")
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with columns: timestamp|ts, price, card_key",
    )
    args = ap.parse_args()

    asyncio.run(run(args.csv))


if __name__ == "__main__":
    main()
