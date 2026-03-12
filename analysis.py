import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib


def convert_parquet_to_csv(source_path, target_path):
    df = pd.read_parquet(source_path)
    df.to_csv(target_path, index=False)
    print("Parquet to CSV conversion complete.")


def filter_noaa_weather_report_by_year_month(source_path, target_path, year, month):
    weather = pd.read_csv(source_path, low_memory=False)
    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")
    weather = weather[["DATE", "PRCP", "TMIN", "TMAX"]]
    weather["PRCP"] = pd.to_numeric(weather["PRCP"], errors="coerce")
    weather["TMIN"] = pd.to_numeric(weather["TMIN"], errors="coerce")
    weather["TMAX"] = pd.to_numeric(weather["TMAX"], errors="coerce")
    weather["PRCP"] = weather["PRCP"] / 10
    weather["TMIN"] = weather["TMIN"] / 10
    weather["TMAX"] = weather["TMAX"] / 10
    weather = weather.rename(columns={
        "DATE": "date",
        "PRCP": "prcp_mm",
        "TMIN": "tmin_c",
        "TMAX": "tmax_c"
    })
    weather_daily = weather[
        (weather["date"].dt.year == year) &
        (weather["date"].dt.month == month)
    ]
    weather_daily = weather_daily.dropna(subset=["prcp_mm", "tmin_c", "tmax_c"], how="all")
    weather_daily = weather_daily.reset_index(drop=True)
    print("Preview:")
    print(weather_daily.head())
    print("\nRows:", len(weather_daily))
    print("Start date:", weather_daily["date"].min())
    print("End date:", weather_daily["date"].max())
    weather_daily.to_csv(target_path, index=False)
    print("\nCleaned weather data saved to:", target_path)


DATA_DIR = Path(r"./data")
OUTPUT_DIR = Path(r"./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_YELLOW_TRIP = "yellow_tripdata_2025-01.parquet"
INPUT_WEATHER_REPORT = "USW00094728.csv"
INPUT_TAXI_LOOKUP = "taxi_zone_lookup.csv"

convert_parquet_to_csv(DATA_DIR / INPUT_YELLOW_TRIP, DATA_DIR / "yellow_tripdata_2025-01.csv")

filter_noaa_weather_report_by_year_month(
    source_path=DATA_DIR / INPUT_WEATHER_REPORT,
    target_path=DATA_DIR / "weather_daily_2025-01.csv",
    year=2025,
    month=1
)

TAXI_FILES = ["yellow_tripdata_2025-01.csv"]
STUDENT_ID = "25118692"
SALT = ""

dfs = []
for f in TAXI_FILES:
    df = pd.read_csv(DATA_DIR / f, low_memory=False)
    dfs.append(df)

trips = pd.concat(dfs, ignore_index=True)
raw_rows = len(trips)
print("Raw rows:", raw_rows)

required_cols = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
    "passenger_count", "fare_amount", "total_amount", "payment_type",
    "PULocationID", "DOLocationID"
]
missing = set(required_cols) - set(trips.columns)
assert not missing, f"Missing columns: {missing}"

trips["pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"], errors="coerce")
trips["dropoff_datetime"] = pd.to_datetime(trips["tpep_dropoff_datetime"], errors="coerce")
trips = trips.dropna(subset=["pickup_datetime", "dropoff_datetime"])
trips = trips[trips["pickup_datetime"] < trips["dropoff_datetime"]]
trips = trips[trips["trip_distance"] > 0]
trips = trips[(trips["fare_amount"] >= 0) & (trips["total_amount"] >= 0)]

Q1 = trips["trip_distance"].quantile(0.25)
Q3 = trips["trip_distance"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 3 * IQR
trips = trips[trips["trip_distance"] <= upper]

trips.loc[(trips["passenger_count"] <= 0) | (trips["passenger_count"] > 8), "passenger_count"] = np.nan

trips["trip_minutes"] = (trips["dropoff_datetime"] - trips["pickup_datetime"]).dt.total_seconds() / 60
trips = trips[trips["trip_minutes"] > 0]
trips["speed_mph"] = trips["trip_distance"] / (trips["trip_minutes"] / 60)
trips = trips[trips["speed_mph"] < 120]
trips["pickup_date"] = trips["pickup_datetime"].dt.normalize()
trips["pickup_hour"] = trips["pickup_datetime"].dt.hour
trips["passenger_count"] = trips.groupby("pickup_hour")["passenger_count"].transform(lambda x: x.fillna(x.median()))

clean_rows = len(trips)

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
print(trips.info())
print(summary.head(10))

plt.hist(trips["speed_mph"], bins=50)
plt.xlabel("Speed MPH")
plt.ylabel("Trips")
plt.savefig(OUTPUT_DIR / "task1_speed_hist.png")
plt.show()
plt.close()

hourly = trips.groupby("pickup_hour").size()
plt.bar(hourly.index, hourly.values)
plt.xlabel("Pickup Hour")
plt.ylabel("Trips")
plt.savefig(OUTPUT_DIR / "task1_hourly_volume.png")
plt.show()
plt.close()

print("Task 1 complete")

zones = pd.read_csv(DATA_DIR / INPUT_TAXI_LOOKUP)
trips = trips.merge(zones.add_prefix("PU_"), left_on="PULocationID", right_on="PU_LocationID", how="left")
trips = trips.merge(zones.add_prefix("DO_"), left_on="DOLocationID", right_on="DO_LocationID", how="left")

od = trips.groupby(["PU_Borough", "DO_Borough"]).agg(
    trip_count=("total_amount", "size"),
    median_total_amount=("total_amount", "median"),
    p90_trip_minutes=("trip_minutes", lambda x: x.quantile(0.9))
).reset_index()
od["share"] = od.groupby("PU_Borough")["trip_count"].transform(lambda x: x / x.sum())
hhi = od.groupby("PU_Borough")["share"].apply(lambda x: (x ** 2).sum())
top_routes = od.sort_values(["PU_Borough", "trip_count"], ascending=[True, False]).groupby("PU_Borough").head(3)
top_routes.to_csv(OUTPUT_DIR / "task2_top_routes.csv", index=False)

plt.bar(hhi.index, hhi.values)
plt.xticks(rotation=45)
plt.ylabel("HHI")
plt.savefig(OUTPUT_DIR / "task2_hhi_bar.png")
plt.show()
plt.close()

print("Task 2 complete")

weather_daily = pd.read_csv(DATA_DIR / "weather_daily_2025-01.csv")
weather_daily["date"] = pd.to_datetime(weather_daily["date"])

trips_weather = trips.merge(weather_daily, left_on="pickup_date", right_on="date", how="left")

taxi_days = trips["pickup_date"].nunique()
weather_days = weather_daily["date"].nunique()
missing_days = trips_weather[trips_weather["date"].isna()]["pickup_date"].astype(str).unique()
join_coverage = (trips_weather["date"].notna().sum() / len(trips_weather)) * 100

join_quality = pd.DataFrame([{
    "taxi_days": taxi_days,
    "weather_days": weather_days,
    "join_coverage": join_coverage,
    "missing_weather_days": ",".join(missing_days)
}])
join_quality.to_csv(OUTPUT_DIR / "task3_weather_join_quality.csv", index=False)
print("Join Quality Report")
print(join_quality)

threshold = weather_daily[weather_daily["prcp_mm"] > 0]["prcp_mm"].quantile(0.75)
weather_daily["rainy_day"] = weather_daily["prcp_mm"] > threshold
print("Rain threshold (mm):", threshold)

joined = trips.merge(weather_daily[["date", "rainy_day"]], left_on="pickup_date", right_on="date", how="left")
rain_stats = joined.groupby("rainy_day")[["total_amount", "trip_minutes"]].median()
print("\nRain Impact Comparison")
print(rain_stats)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
rain_labels = ['No Rain', 'Rainy Day']
total_amounts = [rain_stats.loc[False, 'total_amount'], rain_stats.loc[True, 'total_amount']]
trip_mins = [rain_stats.loc[False, 'trip_minutes'], rain_stats.loc[True, 'trip_minutes']]
ax[0].bar(rain_labels, total_amounts, color=['blue', 'red'])
ax[0].set_ylabel('Median Total Amount ($)')
ax[0].set_title('Total Amount: Rainy vs Non-Rainy Days')
ax[0].set_ylim([0, max(total_amounts) * 1.2])
ax[1].bar(rain_labels, trip_mins, color=['blue', 'red'])
ax[1].set_ylabel('Median Trip Minutes')
ax[1].set_title('Trip Minutes: Rainy vs Non-Rainy Days')
ax[1].set_ylim([0, max(trip_mins) * 1.2])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task3_rain_penalty.png", dpi=100)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
rainy_minutes = joined[joined['rainy_day'] == True]['trip_minutes'].dropna()
non_rainy_minutes = joined[joined['rainy_day'] == False]['trip_minutes'].dropna()
bp = ax.boxplot([non_rainy_minutes, rainy_minutes], tick_labels=['No Rain', 'Rainy Day'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax.set_ylabel('Trip Minutes')
ax.set_title('Distribution of Trip Duration: Rainy vs Non-Rainy Days')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task3_rain_distribution.png", dpi=100)
plt.show()
plt.close()

print("Task 3 complete")


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
anomalies[["pickup_datetime", "dropoff_datetime", "PU_Borough", "DO_Borough", "speed_mph", "rz", "total_amount"]].to_csv(OUTPUT_DIR / "task4_anomalies.csv", index=False)
print(anomalies.head(10))

plt.scatter(trips["trip_distance"], trips["trip_minutes"], s=1)
plt.scatter(anomalies["trip_distance"], anomalies["trip_minutes"], color="red", s=3)
plt.xlabel("Trip Distance")
plt.ylabel("Trip Minutes")
plt.savefig(OUTPUT_DIR / "task4_anomaly_scatter.png")
plt.show()
plt.close()

print("Task 4 complete")

seed = int(hashlib.sha256((STUDENT_ID + SALT).encode()).hexdigest()[:8], 16)
rng = np.random.default_rng(seed)
sample = trips.sample(n=2000, random_state=rng)
cols = ["trip_minutes", "trip_distance", "total_amount", "speed_mph"]
sample = sample[cols].round(3)
means = sample.mean().values
stds = sample.std().values
sig = np.dot(means, [1, 10, 100, 1000]) + np.dot(stds, [2, 20, 200, 2000])
fingerprint = hashlib.sha256(str(sig).encode()).hexdigest()
print("Fingerprint:", fingerprint)

with open(OUTPUT_DIR / "task5_fingerprint.txt", "w") as f:
    f.write(f"STUDENT_ID: {STUDENT_ID}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"clean_rows: {len(trips)}\n")
    f.write(f"sig: {sig}\n")
    f.write(f"fingerprint: {fingerprint}\n")

print("Task 5 complete")
print("All tasks finished successfully")