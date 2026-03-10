import pandas as pd

df = pd.read_parquet("Programming in AI/CA/project/data/yellow_tripdata_2025-01.parquet")
df.to_csv("Programming in AI/CA/project/data/yellow_tripdata_2025-01.csv", index=False)