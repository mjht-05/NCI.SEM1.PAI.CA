# ==========================================================
# Programming for AI – Data Wrangling & Analysis Assignment
# Full pipeline: Tasks 1–5
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib

# -------------------------------
# Configuration
# -------------------------------

DATA_DIR = Path(r"Programming in AI/CA/project/data")
OUTPUT_DIR = Path(r"Programming in AI/CA/project/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TAXI_FILES = [
    "yellow_tripdata_2025-01.csv"
]

STUDENT_ID = "123456789"  # Replace with your actual student ID
SALT = "AI_CA1"

# ==========================================================
# Task 1 — Build trustworthy trip table
# ==========================================================

print("Loading taxi data...")

dfs = []

for f in TAXI_FILES:
    df = pd.read_csv(DATA_DIR / f, low_memory=False)
    dfs.append(df)

trips = pd.concat(dfs, ignore_index=True)

raw_rows = len(trips)
print("Raw rows:", raw_rows)

# -------------------------------
# Schema check
# -------------------------------

required_cols = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
    "payment_type",
    "PULocationID",
    "DOLocationID"
]

missing = set(required_cols) - set(trips.columns)
assert not missing, f"Missing columns: {missing}"

# -------------------------------
# Datetime conversion
# -------------------------------

trips["pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"], errors="coerce")
trips["dropoff_datetime"] = pd.to_datetime(trips["tpep_dropoff_datetime"], errors="coerce")

trips = trips.dropna(subset=["pickup_datetime", "dropoff_datetime"])

# -------------------------------
# Cleaning rules
# -------------------------------

trips = trips[trips["pickup_datetime"] < trips["dropoff_datetime"]]
trips = trips[trips["trip_distance"] > 0]
trips = trips[(trips["fare_amount"] >= 0) & (trips["total_amount"] >= 0)]

# Robust distance rule (IQR)

Q1 = trips["trip_distance"].quantile(0.25)
Q3 = trips["trip_distance"].quantile(0.75)
IQR = Q3 - Q1

upper = Q3 + 3 * IQR

trips = trips[trips["trip_distance"] <= upper]

# Passenger cleanup

trips.loc[
    (trips["passenger_count"] <= 0) | (trips["passenger_count"] > 8),
    "passenger_count"
] = np.nan

# -------------------------------
# Derived columns
# -------------------------------

trips["trip_minutes"] = (
    trips["dropoff_datetime"] - trips["pickup_datetime"]
).dt.total_seconds() / 60

trips["speed_mph"] = trips["trip_distance"] / (trips["trip_minutes"] / 60)

# IMPORTANT FIX
trips["pickup_date"] = trips["pickup_datetime"].dt.normalize()

trips["pickup_hour"] = trips["pickup_datetime"].dt.hour

# Impute passenger count

trips["passenger_count"] = trips.groupby("pickup_hour")["passenger_count"].transform(
    lambda x: x.fillna(x.median())
)

clean_rows = len(trips)

# -------------------------------
# Summary output
# -------------------------------

summary = pd.DataFrame([{
    "raw_rows": raw_rows,
    "clean_rows": clean_rows,
    "percent_dropped": 100 * (raw_rows - clean_rows) / raw_rows,
    "median_trip_minutes": trips["trip_minutes"].median(),
    "median_speed_mph": trips["speed_mph"].median(),
    "share_cash": (trips["payment_type"] == 2).mean(),
    "share_card": (trips["payment_type"] == 1).mean()
}])

summary.to_csv(OUTPUT_DIR / "task1_summary.csv", index=False)

# -------------------------------
# Plots
# -------------------------------

plt.hist(trips["speed_mph"], bins=50)
plt.xlabel("Speed MPH")
plt.ylabel("Trips")
plt.savefig(OUTPUT_DIR / "task1_speed_hist.png")
plt.close()

hourly = trips.groupby("pickup_hour").size()

plt.bar(hourly.index, hourly.values)
plt.xlabel("Pickup Hour")
plt.ylabel("Trips")
plt.savefig(OUTPUT_DIR / "task1_hourly_volume.png")
plt.close()

print("Task 1 complete")

# ==========================================================
# Task 2 — Borough OD flows
# ==========================================================

zones = pd.read_csv(DATA_DIR / "taxi_zone_lookup.csv")

trips = trips.merge(
    zones.add_prefix("PU_"),
    left_on="PULocationID",
    right_on="PU_LocationID",
    how="left"
)

trips = trips.merge(
    zones.add_prefix("DO_"),
    left_on="DOLocationID",
    right_on="DO_LocationID",
    how="left"
)

od = trips.groupby(["PU_Borough", "DO_Borough"]).agg(
    trip_count=("total_amount", "size"),
    median_total_amount=("total_amount", "median"),
    p90_trip_minutes=("trip_minutes", lambda x: x.quantile(0.9))
).reset_index()

od["share"] = od.groupby("PU_Borough")["trip_count"].transform(lambda x: x / x.sum())

hhi = od.groupby("PU_Borough")["share"].apply(lambda x: (x ** 2).sum())

plt.bar(hhi.index, hhi.values)
plt.xticks(rotation=45)
plt.ylabel("HHI")
plt.savefig(OUTPUT_DIR / "task2_hhi_bar.png")
plt.close()

print("Task 2 complete")

# ==========================================================
# Task 3 — Weather join
# ==========================================================

weather = pd.read_csv(DATA_DIR / "weather_daily_2025-01.csv")

# Fix column names if raw NOAA format used
weather.rename(columns={
    "DATE": "date",
    "PRCP": "prcp_mm",
    "TMIN": "tmin_c",
    "TMAX": "tmax_c"
}, inplace=True)

weather["date"] = pd.to_datetime(weather["date"])

trips_weather = trips.merge(
    weather,
    left_on="pickup_date",
    right_on="date",
    how="left"
)

# Rain threshold

threshold = weather["prcp_mm"][weather["prcp_mm"] > 0].quantile(0.75)

weather["rainy_day"] = weather["prcp_mm"] > threshold

joined = trips.merge(
    weather[["date", "rainy_day"]],
    left_on="pickup_date",
    right_on="date",
    how="left"
)

rain_stats = joined.groupby("rainy_day")[["total_amount", "trip_minutes"]].median()

print("Rain impact")
print(rain_stats)

print("Task 3 complete")

# ==========================================================
# Task 4 — Robust anomaly detection
# ==========================================================

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros(len(x))
    return (x - med) / (1.4826 * mad)

groups = ["pickup_hour", "PU_Borough"]

anoms = []

for name, group in trips.groupby(groups):

    if len(group) < 50:
        continue

    rz = robust_z(group["speed_mph"])

    group = group.copy()
    group["rz"] = rz

    anoms.append(group[np.abs(rz) > 6])

anomalies = pd.concat(anoms)

anomalies[[
    "pickup_datetime",
    "dropoff_datetime",
    "PU_Borough",
    "DO_Borough",
    "speed_mph",
    "rz",
    "total_amount"
]].to_csv(OUTPUT_DIR / "task4_anomalies.csv", index=False)

plt.scatter(trips["trip_distance"], trips["trip_minutes"], s=1)

plt.scatter(
    anomalies["trip_distance"],
    anomalies["trip_minutes"],
    color="red",
    s=3
)

plt.xlabel("Trip Distance")
plt.ylabel("Trip Minutes")

plt.savefig(OUTPUT_DIR / "task4_anomaly_scatter.png")
plt.close()

print("Task 4 complete")

# ==========================================================
# Task 5 — Deterministic fingerprint
# ==========================================================

seed = int(
    hashlib.sha256((STUDENT_ID + SALT).encode()).hexdigest()[:8],
    16
)

rng = np.random.default_rng(seed)

sample = trips.sample(n=2000, random_state=rng)

cols = ["trip_minutes", "trip_distance", "total_amount", "speed_mph"]

sample = sample[cols].round(3)

means = sample.mean().values
stds = sample.std().values

sig = np.dot(means, [1, 10, 100, 1000]) + np.dot(stds, [2, 20, 200, 2000])

fingerprint = hashlib.sha256(str(sig).encode()).hexdigest()

with open(OUTPUT_DIR / "task5_fingerprint.txt", "w") as f:

    f.write(f"STUDENT_ID: {STUDENT_ID}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"clean_rows: {len(trips)}\n")
    f.write(f"sig: {sig}\n")
    f.write(f"fingerprint: {fingerprint}\n")

print("Task 5 complete")
print("All tasks finished successfully")