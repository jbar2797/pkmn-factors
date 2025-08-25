from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Sequence

from pkmn_factors.universe import load_universe


async def _run_cmd(args: Sequence[str]) -> int:
    """Run a subprocess and return a definite int return code (mypy-safe)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    if stdout:
        try:
            print(stdout.decode("utf-8"), end="")
        except Exception:
            print(stdout)
    # After communicate(), returncode is set but typed Optional[int].
    # Wait returns a concrete int; this silences mypy correctly.
    returncode: int = await proc.wait()
    return returncode


async def run(
    universe_path: Path,
    model_version: str,
    horizon_days: int,
    note: Optional[str],
) -> None:
    keys = sorted(load_universe(universe_path))
    if not keys:
        print(f"[run_universe] No keys in universe file: {universe_path}")
        return

    for key in keys:
        print(f"\n=== [{key}] generate signal ({model_version}) ===")
        rc = await _run_cmd(
            [
                sys.executable,
                "-m",
                f"pkmn_factors.models.{model_version}",
                "--card-key",
                key,
            ]
        )
        if rc != 0:
            print(f"[WARN] model run failed for {key} (rc={rc}); continuing")
            continue

        print(f"=== [{key}] backtest & persist metrics ===")
        bt_args = [
            sys.executable,
            "-m",
            "pkmn_factors.eval.backtest",
            "--card-key",
            key,
            "--model-version",
            model_version,
            "--horizon-days",
            str(horizon_days),
            "--persist",
        ]
        if note:
            bt_args += ["--note", note]
        rc = await _run_cmd(bt_args)
        if rc != 0:
            print(f"[WARN] backtest failed for {key} (rc={rc}); continuing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--universe-path",
        type=Path,
        default=Path("data/universe_demo.csv"),
        help="CSV listing card_key,enabled",
    )
    ap.add_argument(
        "--model-version",
        default="bhs_baseline_v2",
        help="Model module name under pkmn_factors.models.*",
    )
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    asyncio.run(
        run(
            args.universe_path,
            args.model_version,
            args.horizon_days,
            args.note or None,
        )
    )


if __name__ == "__main__":
    main()
