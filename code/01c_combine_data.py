#!/usr/bin/env python3
"""Combine all collected station JSON files into a single DataFrame."""
import os, json, pickle
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RAW_DIR = DATA_DIR / "raw_stations"

# Load station metadata
with open(DATA_DIR / "stations.json") as f:
    stations = json.load(f)
stations_meta = {s["stationName"]: s for s in stations}

# Combine all station files
print("Combining all station data...")
all_records = []
station_files = sorted(RAW_DIR.glob("*.json"))
print(f"Found {len(station_files)} station files")

for sf in station_files:
    with open(sf) as f:
        items = json.load(f)
    all_records.extend(items)

print(f"Total records: {len(all_records)}")

df = pd.DataFrame(all_records)

# Convert numeric columns
numeric_cols = ["pm25Value", "pm10Value", "so2Value", "no2Value", "coValue", "o3Value",
                "pm25Value24", "pm10Value24"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Parse datetime
df["dataTime"] = pd.to_datetime(df["dataTime"], format="%Y-%m-%d %H:%M", errors="coerce")

# Add lat/lon and addr from station metadata
df["lat"] = df["stationName"].map(lambda x: float(stations_meta.get(x, {}).get("dmX", "0")) if stations_meta.get(x, {}).get("dmX") else np.nan)
df["lon"] = df["stationName"].map(lambda x: float(stations_meta.get(x, {}).get("dmY", "0")) if stations_meta.get(x, {}).get("dmY") else np.nan)
df["addr"] = df["stationName"].map(lambda x: stations_meta.get(x, {}).get("addr", ""))
df["item_list"] = df["stationName"].map(lambda x: stations_meta.get(x, {}).get("item", ""))

# Extract sido from addr
df["sido"] = df["addr"].str.split().str[0]

# Sort by station and time
df = df.sort_values(["stationName", "dataTime"]).reset_index(drop=True)

print(f"\nFinal DataFrame shape: {df.shape}")
print(f"Date range: {df['dataTime'].min()} to {df['dataTime'].max()}")
print(f"Stations: {df['stationName'].nunique()}")
print(f"\nStation types:")
print(df["mangName"].value_counts().to_string())
print(f"\nSido distribution:")
print(df["sido"].value_counts().head(10).to_string())
print(f"\nMissing values per sensor:")
for col in ["pm25Value", "pm10Value", "so2Value", "no2Value", "coValue", "o3Value"]:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        print(f"  {col}: {pct:.1f}%")

# Save
outpath = DATA_DIR / "airquality_raw.pkl"
df.to_pickle(outpath)
print(f"\nSaved to {outpath}")
print(f"File size: {outpath.stat().st_size / 1024 / 1024:.1f} MB")
