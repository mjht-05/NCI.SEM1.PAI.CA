import pandas as pd
import numpy as np

# ------------------------------------
# 1. Load NOAA weather dataset
# ------------------------------------
weather = pd.read_csv("Programming in AI\CA\project\data\org_noaa_weather_station.csv", low_memory=False)

# ------------------------------------
# 2. Convert DATE column
# ------------------------------------
weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")

# ------------------------------------
# 3. Keep only columns needed for assignment
# ------------------------------------
weather = weather[["DATE", "PRCP", "TMIN", "TMAX"]]

# ------------------------------------
# 4. Convert numeric values safely
# ------------------------------------
weather["PRCP"] = pd.to_numeric(weather["PRCP"], errors="coerce")
weather["TMIN"] = pd.to_numeric(weather["TMIN"], errors="coerce")
weather["TMAX"] = pd.to_numeric(weather["TMAX"], errors="coerce")

# ------------------------------------
# 5. Convert NOAA scaled units
# PRCP -> millimetres
# TMIN/TMAX -> Celsius
# ------------------------------------
weather["PRCP"] = weather["PRCP"] / 10
weather["TMIN"] = weather["TMIN"] / 10
weather["TMAX"] = weather["TMAX"] / 10

# ------------------------------------
# 6. Rename columns as required by assignment
# ------------------------------------
weather = weather.rename(columns={
    "DATE": "date",
    "PRCP": "prcp_mm",
    "TMIN": "tmin_c",
    "TMAX": "tmax_c"
})

# ------------------------------------
# 7. Filter required months
# (Oct, Nov, Dec 2025 and Jan 2026)
# ------------------------------------
weather_daily = weather[
    (
        (weather["date"].dt.year == 2025) &
        (weather["date"].dt.month == 1)
    )
]

# ------------------------------------
# 8. Remove rows with no weather data
# ------------------------------------
weather_daily = weather_daily.dropna(
    subset=["prcp_mm", "tmin_c", "tmax_c"],
    how="all"
)

# ------------------------------------
# 9. Reset index
# ------------------------------------
weather_daily = weather_daily.reset_index(drop=True)

# ------------------------------------
# 10. Check results
# ------------------------------------
print(weather_daily.head())
print(weather_daily.tail())

print("Rows:", len(weather_daily))
print("Start date:", weather_daily["date"].min())
print("End date:", weather_daily["date"].max())

weather_daily.to_csv("Programming in AI\CA\project\data\weather_daily_2025-01.csv", index=False)
print("Cleaned weather data saved.")