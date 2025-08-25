from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from pkmn_factors.ingest.csv_to_trades import ingest_csv

DEFAULT_CSV = "data/trades_demo.csv"


async def _run(
    csv_path: str | Path,
    card_key: str,
    model_version: str,
    horizon_days: int,
    note: str,
) -> None:
    # 1) Ingest CSV (force source/currency; override card_key to match model/backtest)
    inserted = await ingest_csv(
        csv_path,
        source="demo_csv",
        card_key_override=card_key,  # <- this is the correct name
        currency="USD",
    )
    print(f"Ingested {inserted} rows from {csv_path} (card_key={card_key})")

    # 2) Run the model to generate signals
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pkmn_factors.models.bhs_baseline_v2",
            "--card-key",
            card_key,
        ],
        check=True,
    )

    # 3) Backtest (print-only)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pkmn_factors.eval.backtest",
            "--card-key",
            card_key,
            "--model-version",
            model_version,
            "--horizon-days",
            str(horizon_days),
        ],
        check=True,
    )

    # 4) Backtest again and persist a metrics row
    cmd = [
        sys.executable,
        "-m",
        "pkmn_factors.eval.backtest",
        "--card-key",
        card_key,
        "--model-version",
        model_version,
        "--horizon-days",
        str(horizon_days),
        "--persist",
    ]
    if note:
        cmd += ["--note", note]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--card-key", required=True)
    ap.add_argument("--model-version", type=str, default="bhs_baseline_v2")
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    asyncio.run(
        _run(
            csv_path=args.csv,
            card_key=args.card_key,
            model_version=args.model_version,
            horizon_days=args.horizon_days,
            note=args.note,
        )
    )


if __name__ == "__main__":
    main()
