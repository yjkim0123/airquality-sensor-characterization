#!/usr/bin/env python3
"""
01_collect_data.py - Collect 3 months of air quality data for all 672 stations.
Uses the realtime API with dataTerm=3MONTH.
Saves to data/airquality_raw.pkl
"""
import os, sys, json, time, pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RAW_DIR = DATA_DIR / "raw_stations"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "41de1ce8903be5cd834094374a879625836ac3743c641b7886f3628ac3c697ef"
URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

# Load station list
with open(DATA_DIR / "stations.json") as f:
    stations = json.load(f)

print(f"Total stations: {len(stations)}")

# Filter out already collected
to_collect = []
for st in stations:
    name = st["stationName"]
    outfile = RAW_DIR / f"{name}.json"
    if not outfile.exists():
        to_collect.append(st)

print(f"Already collected: {len(stations) - len(to_collect)}")
print(f"Remaining: {len(to_collect)}")

# Rate limiter
rate_lock = threading.Lock()
last_request_time = [0.0]

def fetch_station(st):
    name = st["stationName"]
    outfile = RAW_DIR / f"{name}.json"
    
    # Rate limit: at least 0.3s between requests
    with rate_lock:
        now = time.time()
        wait = 0.3 - (now - last_request_time[0])
        if wait > 0:
            time.sleep(wait)
        last_request_time[0] = time.time()
    
    try:
        params = {
            "serviceKey": API_KEY,
            "returnType": "json",
            "numOfRows": 2500,
            "pageNo": 1,
            "stationName": name,
            "dataTerm": "3MONTH",
            "ver": "1.5"
        }
        r = requests.get(URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        body = data.get("response", {}).get("body", {})
        items = body.get("items", [])
        total = body.get("totalCount", 0)
        
        if items and total > 0:
            with open(outfile, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False)
            return ("ok", name, len(items))
        else:
            return ("nodata", name, str(data)[:200])
    except Exception as e:
        return ("error", name, str(e))

# Use 3 threads to speed up (still respectful)
errors = []
collected = 0
no_data = 0

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(fetch_station, st): st for st in to_collect}
    done_count = 0
    for future in as_completed(futures):
        result = future.result()
        done_count += 1
        if result[0] == "ok":
            collected += 1
        elif result[0] == "nodata":
            no_data += 1
            errors.append({"station": result[1], "error": "no data", "detail": result[2]})
        else:
            errors.append({"station": result[1], "error": result[2]})
        
        if done_count % 50 == 0:
            print(f"  Progress: {done_count}/{len(to_collect)} | Collected: {collected} | Errors: {len(errors)}")

print(f"\nCollection complete: {collected} new, {no_data} no data, {len(errors)} errors")

# Save errors
with open(DATA_DIR / "collection_errors.json", "w") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)

# Now combine all into a single DataFrame
print("\nCombining all station data...")
all_records = []
station_files = list(RAW_DIR.glob("*.json"))
print(f"Found {len(station_files)} station files")

for sf in station_files:
    with open(sf) as f:
        items = json.load(f)
    for item in items:
        item["_station_file"] = sf.stem
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

# Merge station metadata
stations_df = pd.DataFrame(stations)
stations_df = stations_df.rename(columns={"dmX": "lat", "dmY": "lon"})
stations_df["lat"] = pd.to_numeric(stations_df["lat"], errors="coerce")
stations_df["lon"] = pd.to_numeric(stations_df["lon"], errors="coerce")

df = df.merge(stations_df[["stationName", "lat", "lon", "mangName", "addr", "item"]],
              on="stationName", how="left")

# Extract sido from addr
df["sido"] = df["addr"].str.split().str[0]

print(f"\nFinal DataFrame shape: {df.shape}")
print(f"Date range: {df['dataTime'].min()} to {df['dataTime'].max()}")
print(f"Stations: {df['stationName'].nunique()}")
print(f"Station types:\n{df['mangName'].value_counts()}")
print(f"\nMissing values per sensor:")
for col in ["pm25Value", "pm10Value", "so2Value", "no2Value", "coValue", "o3Value"]:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        print(f"  {col}: {pct:.1f}%")

# Save
df.to_pickle(DATA_DIR / "airquality_raw.pkl")
print(f"\nSaved to {DATA_DIR / 'airquality_raw.pkl'}")
print(f"File size: {(DATA_DIR / 'airquality_raw.pkl').stat().st_size / 1024 / 1024:.1f} MB")
