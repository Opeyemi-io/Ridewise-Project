from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / 'models/random_forest.joblib'


model = joblib.load(MODEL_PATH)
model_features = list(getattr(model, "feature_names_in_", []))
if not model_features:
    raise RuntimeError("model has no feature_names_in_; check the saved artifact")
    
app = FastAPI(title="RideWise Churn Scoring API")


class Riders(BaseModel):
    model_config = {"extra": "forbid"}

    # Raw rider inputs.
    avg_rating_given: float = Field(..., json_schema_extra={"example": 4.6})
    riders_referred: int = Field(..., json_schema_extra={"example": 2})
    was_referred: bool = Field(..., json_schema_extra={"example": True})
    loyalty_status: str = Field(..., json_schema_extra={"example": "Gold"})
    city: str = Field(..., json_schema_extra={"example": "Lagos"})

    # Historical or aggregate inputs from trips and sessions.
    total_trips: int = Field(..., json_schema_extra={"example": 18})
    total_spent: float = Field(..., json_schema_extra={"example": 120.5})
    fare_std: float = Field(..., json_schema_extra={"example": 3.2})
    max_fare: float = Field(..., json_schema_extra={"example": 12.0})
    total_tip: float = Field(..., json_schema_extra={"example": 8.5})
    tip_rate: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.44})
    avg_surge: float = Field(..., json_schema_extra={"example": 1.05})
    max_surge: float = Field(..., json_schema_extra={"example": 1.8})
    surge_trip_ratio: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.22})
    avg_duration: float = Field(..., json_schema_extra={"example": 14.5})
    duration_std: float = Field(..., json_schema_extra={"example": 4.1})
    avg_time_on_app: float = Field(..., json_schema_extra={"example": 11.3})
    time_on_app_std: float = Field(..., json_schema_extra={"example": 2.1})
    conversion_rate: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.37})

    # Engineered features from the leak-free RideWise pipeline.
    weekend_ratio: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.31})
    peak_hour_ratio: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.52})
    tenure_days: int = Field(..., json_schema_extra={"example": 120})
    trips_per_month: float = Field(..., json_schema_extra={"example": 4.5})
    recency_at_cutoff: float = Field(..., json_schema_extra={"example": 6})
    spend_per_trip: float = Field(..., json_schema_extra={"example": 6.69})
    trips_per_session: float = Field(..., json_schema_extra={"example": 1.8})
    session_trip_gap: float = Field(..., json_schema_extra={"example": 2})
@app.post("/predict")
def predict_churn(transaction: Riders):
    features = pd.get_dummies(pd.DataFrame([transaction.model_dump()]), columns=["loyalty_status", "city"], dtype=int).reindex(columns=model_features, fill_value=0)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "label": "churn" if prediction == 1 else "retained",
    }
