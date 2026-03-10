Datasets used:

NYC Taxi Yellow Trip Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet

Taxi Zone Lookup
https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

NOAA Weather Data
https://www.ncei.noaa.gov/


# -------------------------------
# Configuration
# Download parquet files for January 2025 yellow taxi data from:
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
# Rename to yellow_tripdata_2025-01.parquet and place in data folder


# Download below csv and dump into the data folder:
# https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv
# Rename it to org_noaa_weather_station.csv
# Run cleanup_weather.py to get weather_daily_2025-01.csv with columns: DATE, PRCP, TMIN, TMAX

# Download csv and dump into the data folder:
# https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

# Run clean_yellow_trip.py to get the parquet files for task 1
# -------------------------------





