from __future__ import annotations

import asyncio
from pathlib import Path

from pkmn_factors.ingest.csv_to_trades import ingest_csv
from pkmn_factors.models.bhs_baseline_v2 import run as run_model
from pkmn_factors.eval.backtest import run as run_backtest


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/trades_demo.csv"))
    ap.add_argument("--card-key", type=str, default="mew-ex-053-svp-2023")
    ap.add_argument("--model-version", type=str, default="bhs_baseline_v2")
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--note", type=str, default="demo pipeline")
    args = ap.parse_args()

    async def _run() -> None:
        # 1) Ingest demo CSV (inject source and override card key so everything lines up)
        inserted = await ingest_csv(
            args.csv,
            source="demo_csv",
            override_card_key=args.card_key,
        )
        print(f"Ingested {inserted} rows from {args.csv} (card_key={args.card_key})")

        # 2) Run the model to generate/refresh a signal
        await run_model(args.card_key)

        # 3) Run backtest and persist a metrics row
        await run_backtest(
            args.card_key,
            args.model_version,
            args.horizon_days,
            persist=True,
            note=args.note,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
