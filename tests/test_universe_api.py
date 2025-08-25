# tests/test_universe_api.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi.testclient import TestClient

import pkmn_factors.api.universe as universe
from pkmn_factors.api.main import app


def _sample_rows():
    return [
        universe.TopMetric(
            card_key="mew-ex-053-svp-2023",
            model_version="bhs_baseline_v2",
            horizon_days=90,
            cum_return=0.1234,
            sharpe=1.5,
            asof_ts=datetime(2025, 8, 24, tzinfo=timezone.utc),
        ),
        universe.TopMetric(
            card_key="other-card",
            model_version="bhs_baseline_v2",
            horizon_days=90,
            cum_return=0.0101,
            sharpe=0.1,
            asof_ts=datetime(2025, 8, 23, tzinfo=timezone.utc),
        ),
    ]


def test_universe_top_endpoint(monkeypatch):
    async def fake_fetch(
        limit: int, model_version, horizon_days
    ) -> List[universe.TopMetric]:
        return _sample_rows()[:limit]

    monkeypatch.setattr(universe, "_fetch_top", fake_fetch)

    client = TestClient(app)
    r = client.get("/universe/top?limit=1")
    assert r.status_code == 200
    js = r.json()
    assert isinstance(js, list) and len(js) == 1
    assert js[0]["card_key"] == "mew-ex-053-svp-2023"


def test_dashboard_html(monkeypatch):
    async def fake_fetch(limit: int, model_version, horizon_days):
        return _sample_rows()[:limit]

    monkeypatch.setattr(universe, "_fetch_top", fake_fetch)

    client = TestClient(app)
    r = client.get("/dashboard?limit=2")
    assert r.status_code == 200
    assert "Universe snapshot" in r.text
    assert "mew-ex-053-svp-2023" in r.text
