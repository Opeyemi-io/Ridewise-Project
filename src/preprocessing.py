from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ridewise project dataset"
RIDERS_PATH = DATA_DIR / "riders.csv"
TRIPS_PATH = DATA_DIR / "trips.csv"
SESSIONS_PATH = DATA_DIR / "sessions.csv"
OUTPUT_PATH = DATA_DIR / "model_input.parquet"

FEATURE_COLUMNS = [
	"loyalty_status",
	"city",
	"avg_rating_given",
	"riders_referred",
	"was_referred",
	"total_trips",
	"total_spent",
	"fare_std",
	"max_fare",
	"total_tip",
	"tip_rate",
	"avg_surge",
	"max_surge",
	"surge_trip_ratio",
	"avg_duration",
	"duration_std",
	"weekend_ratio",
	"peak_hour_ratio",
	"tenure_days",
	"trips_per_month",
	"recency_at_cutoff",
	"spend_per_trip",
	"avg_time_on_app",
	"time_on_app_std",
	"conversion_rate",
	"trips_per_session",
	"session_trip_gap",
]

DROP_FEATURES = ["avg_fare", "avg_tip", "total_sessions", "avg_pages_visited"]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	riders = pd.read_csv(RIDERS_PATH, dtype={"user_id": "string"})
	trips = pd.read_csv(TRIPS_PATH, dtype={"trip_id": "string", "user_id": "string", "driver_id": "string"})
	sessions = pd.read_csv(SESSIONS_PATH, dtype={"session_id": "string", "rider_id": "string"})

	riders["signup_date"] = pd.to_datetime(riders["signup_date"], errors="coerce")
	trips["pickup_time"] = pd.to_datetime(trips["pickup_time"], errors="coerce", utc=True)
	trips["dropoff_time"] = pd.to_datetime(trips["dropoff_time"], errors="coerce", utc=True)
	sessions["session_time"] = pd.to_datetime(sessions["session_time"], errors="coerce", utc=True)

	for frame in (trips, sessions):
		for column in frame.columns:
			if frame[column].dtype == object:
				frame[column] = frame[column].replace({"": np.nan})

	return riders, trips, sessions


def build_rider_features(riders: pd.DataFrame) -> pd.DataFrame:
	rider_features = riders[["user_id", "loyalty_status", "city", "avg_rating_given", "referred_by"]].copy()
	rider_features["was_referred"] = rider_features["referred_by"].notna().astype(int)

	referred_counts = (
		riders.dropna(subset=["referred_by"])
		.groupby("referred_by")
		.size()
		.rename("riders_referred")
	)
	rider_features["riders_referred"] = rider_features["user_id"].map(referred_counts).fillna(0).astype(int)

	return rider_features.drop(columns=["referred_by"])


def _prepare_trip_frame(trips: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
	trip_frame = trips.loc[trips["pickup_time"] <= cutoff_date].copy()
	trip_frame["hour"] = trip_frame["pickup_time"].dt.hour
	trip_frame["is_weekend"] = (trip_frame["pickup_time"].dt.dayofweek >= 5).astype(int)
	trip_frame["is_peak"] = trip_frame["hour"].between(7, 9) | trip_frame["hour"].between(17, 19)
	trip_frame["trip_duration"] = (trip_frame["dropoff_time"] - trip_frame["pickup_time"]).dt.total_seconds() / 60
	return trip_frame


def build_trip_features(trips: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
	trips_past = _prepare_trip_frame(trips, cutoff_date)

	trip_features = trips_past.groupby("user_id").agg(
		total_trips=("trip_id", "count"),
		total_spent=("fare", "sum"),
		avg_fare=("fare", "mean"),
		fare_std=("fare", "std"),
		max_fare=("fare", "max"),
		total_tip=("tip", "sum"),
		avg_tip=("tip", "mean"),
		tip_rate=("tip", lambda values: (values > 0).mean()),
		avg_surge=("surge_multiplier", "mean"),
		max_surge=("surge_multiplier", "max"),
		surge_trip_ratio=("surge_multiplier", lambda values: (values > 1).mean()),
		avg_duration=("trip_duration", "mean"),
		duration_std=("trip_duration", "std"),
		weekend_ratio=("is_weekend", "mean"),
		peak_hour_ratio=("is_peak", "mean"),
		first_trip=("pickup_time", "min"),
		last_trip=("pickup_time", "max"),
	).reset_index()

	trip_features["tenure_days"] = (trip_features["last_trip"] - trip_features["first_trip"]).dt.days
	trip_features["trips_per_month"] = trip_features["total_trips"] / ((trip_features["tenure_days"] / 30) + 1)
	trip_features["recency_at_cutoff"] = (cutoff_date - trip_features["last_trip"]).dt.days
	trip_features["spend_per_trip"] = trip_features["total_spent"] / (trip_features["total_trips"] + 1)

	return trip_features.drop(columns=["first_trip", "last_trip"])


def build_session_features(sessions: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
	sessions_past = sessions.loc[sessions["session_time"] <= cutoff_date].copy()

	session_features = sessions_past.groupby("rider_id").agg(
		total_sessions=("session_id", "count"),
		avg_time_on_app=("time_on_app", "mean"),
		time_on_app_std=("time_on_app", "std"),
		avg_pages_visited=("pages_visited", "mean"),
		total_conversions=("converted", "sum"),
		conversion_rate=("converted", "mean"),
	).reset_index().rename(columns={"rider_id": "user_id"})

	return session_features


def build_labels(riders: pd.DataFrame, trips: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
	active_after = set(trips.loc[trips["pickup_time"] > cutoff_date, "user_id"].dropna().unique())
	labels = pd.DataFrame({"user_id": riders["user_id"].dropna().unique()})
	labels["churn"] = (~labels["user_id"].isin(active_after)).astype(int)
	return labels


def build_model_input() -> pd.DataFrame:
	riders, trips, sessions = load_raw_data()
	reference_date = trips["pickup_time"].max()
	cutoff_date = reference_date - pd.Timedelta(days=30)

	rider_features = build_rider_features(riders)
	trip_features = build_trip_features(trips, cutoff_date)
	session_features = build_session_features(sessions, cutoff_date)
	labels = build_labels(riders, trips, cutoff_date)

	master = (
		riders[["user_id", "loyalty_status", "city"]]
		.merge(rider_features, on=["user_id", "loyalty_status", "city"], how="left")
		.merge(trip_features, on="user_id", how="left")
		.merge(session_features, on="user_id", how="left")
		.merge(labels, on="user_id", how="left")
	)

	master["loyalty_status"] = master["loyalty_status"].fillna("Unknown")
	master["city"] = master["city"].fillna("Unknown")
	master["churn"] = master["churn"].fillna(0).astype(int)
	master["trips_per_session"] = master["total_trips"] / (master["total_sessions"] + 1)
	master["session_trip_gap"] = master["total_sessions"] - master["total_trips"]

	master = master.drop(columns=DROP_FEATURES, errors="ignore").fillna(0)
	model_df = pd.get_dummies(master[FEATURE_COLUMNS + ["churn"]], columns=["loyalty_status", "city"], dtype=int)

	DATA_DIR.mkdir(parents=True, exist_ok=True)
	model_df.to_parquet(OUTPUT_PATH, index=False)
	return model_df


def main() -> None:
	model_df = build_model_input()
	print(f"Saved {model_df.shape} -> {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
