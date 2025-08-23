from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pkmn_factors.core.scoring_rt import compute_score_rt

app = FastAPI(title="PKMN Factors (RT)")

class ScorePayload(BaseModel):
    row: dict
    trades_csv_path: str | None = None

@app.post("/score")
def score(payload: ScorePayload):
    trades = pd.read_csv(payload.trades_csv_path, parse_dates=["timestamp"]) if payload.trades_csv_path else None
    return compute_score_rt(payload.row, trades)
