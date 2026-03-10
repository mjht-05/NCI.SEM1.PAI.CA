import pandas as pd

df = pd.read_csv("Programming in AI\CA\project\data\yellow_tripdata_2025-01.csv", nrows=10000)
df.to_csv("Programming in AI\CA\project\data\yellow_10000_2025-01.csv", index=False)