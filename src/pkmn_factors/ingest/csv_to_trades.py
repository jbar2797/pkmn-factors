from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# our central engine factory (returns an AsyncEngine when called)
from pkmn_factors.db.base import engine as engine_factory


async def _ingest_csv(csv_path: str) -> None:
    """
    Load a CSV and (placeholder) touch the DB connection.

    This is intentionally minimal so mypy and CI are happy.
    Replace the placeholder SQL with your actual INSERT / COPY.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    # read just to verify the file; you can use `df` below in your real insert
    df = pd.read_csv(p)
    _ = df.shape  # avoid 'unused variable' lints; remove once you use df

    # IMPORTANT: call the factory to get an AsyncEngine instance
    eng: AsyncEngine = engine_factory()

    # simple connectivity check / placeholder write
    async with eng.begin() as conn:
        # TODO: replace this with your real INSERTs into trades table
        await conn.execute(text("SELECT 1"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Ingest trades CSV into database.")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    args = ap.parse_args()

    asyncio.run(_ingest_csv(args.csv))


if __name__ == "__main__":
    main()
