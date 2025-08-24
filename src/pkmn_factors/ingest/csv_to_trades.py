from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from pkmn_factors.db.base import engine
from pkmn_factors.db.models import Trade

USAGE = "Usage: python -m pkmn_factors.ingest.csv_to_trades <path/to/trades.csv>"


async def main(path: str) -> None:
    # --- Load CSV ---
    df = pd.read_csv(path)

    # Ensure required logical columns exist
    if "card_key" not in df.columns:
        df["card_key"] = "DEMO"
    if "currency" not in df.columns:
        df["currency"] = "USD"

    # Parse timestamps to tz-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Ensure required columns exist even if missing in CSV
    required = ["timestamp", "price", "source", "card_key", "currency"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Drop rows lacking minimal required fields
    df = df.dropna(subset=["timestamp", "price", "source"])

    # Convert to list-of-dicts with Python-native types
    rows: List[Dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        ts = r["timestamp"]
        r["timestamp"] = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        rows.append(
            {
                "timestamp": r["timestamp"],
                "source": r.get("source"),
                "card_key": r.get("card_key"),
                "grade": r.get("grade"),
                "listing_type": r.get("listing_type"),
                "price": r.get("price"),
                "currency": r.get("currency", "USD"),
                "link": r.get("link"),
            }
        )

    if not rows:
        print(f"No valid rows found in {path}. Nothing to insert.")
        return

    # --- Build INSERT ... ON CONFLICT DO NOTHING (PostgreSQL dialect) ---
    # Use the table object explicitly for the dialect insert to satisfy typing.
    stmt = (
        pg_insert(Trade.__table__)  # type: ignore[arg-type]
        .values(rows)
        .on_conflict_do_nothing(index_elements=["timestamp", "price", "source", "link"])
    )

    # --- Execute in a single transaction ---
    async with engine.begin() as conn:
        try:
            await conn.execute(stmt)
            print(f"Inserted rows (attempted): {len(rows)}")
        except IntegrityError:
            # Normally avoided by DO NOTHING, but kept for safety.
            print("IntegrityError: duplicates encountered (skipped).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(USAGE)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
