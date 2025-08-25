from __future__ import annotations

import asyncio
import datetime as dt
import subprocess
import sys
from typing import Sequence

from ..eval.backtest import run as run_backtest

# A small universe to start; we can grow this later
CARDS: Sequence[str] = ("mew-ex-053-svp-2023",)

MODEL_VERSION = "bhs_baseline_v2"
HORIZON_DAYS = 90


async def _run_one(card_key: str) -> None:
    # 1) generate a fresh signal with the model's CLI
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

    # 2) backtest + persist a metrics row
    note = f"nightly {dt.datetime.utcnow().isoformat()}Z"
    await run_backtest(card_key, MODEL_VERSION, HORIZON_DAYS, True, note)


async def main() -> None:
    # Sequential is fine at this scale; we can parallelize later
    for ck in CARDS:
        await _run_one(ck)


if __name__ == "__main__":
    asyncio.run(main())
