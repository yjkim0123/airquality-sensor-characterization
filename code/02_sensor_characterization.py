#!/usr/bin/env python3
"""
02_sensor_characterization.py - Sensor noise, drift, failure analysis.
"""
import os, json, pickle, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_pickle(DATA_DIR / "airquality_raw.pkl")
print(f"Shape: {df.shape}, Stations: {df['stationName'].nunique()}")

SENSORS = {
    "pm25Value": "PM$_{2.5}$",
    "pm10Value": "PM$_{10}$",
    "so2Value": "SO$_2$",
    "no2Value": "NO$_2$",
    "coValue": "CO",
    "o3Value": "O$_3$"
}

SENSOR_UNITS = {
    "pm25Value": "μg/m³",
    "pm10Value": "μg/m³",
    "so2Value": "ppm",
    "no2Value": "ppm",
    "coValue": "ppm",
    "o3Value": "ppm"
}

# ============================================================
# 1. NOISE CHARACTERIZATION
# ============================================================
print("\n=== 1. Noise Characterization ===")

noise_results = {}

for col, label in SENSORS.items():
    print(f"\n  Processing {col}...")
    
    # Filter valid data
    valid = df[df[col].notna() & (df[col] > 0)].copy()
    
    # Daily CV per station: for each station-day, compute CV of hourly readings
    valid["date"] = valid["dataTime"].dt.date
    daily_stats = valid.groupby(["stationName", "date"])[col].agg(["mean", "std", "count"])
    daily_stats = daily_stats[daily_stats["count"] >= 12]  # at least 12 hours
    daily_stats["cv"] = daily_stats["std"] / daily_stats["mean"]
    daily_stats = daily_stats[daily_stats["cv"].notna() & np.isfinite(daily_stats["cv"])]
    
    # Station-level CV (mean of daily CVs)
    station_cv = daily_stats.groupby("stationName")["cv"].median()
    
    # Weekly CV per station
    valid = valid[valid["dataTime"].notna()].copy()
    valid["week"] = valid["dataTime"].dt.isocalendar().week.to_numpy(dtype="int64", na_value=0)
    valid["year"] = valid["dataTime"].dt.year
    weekly_stats = valid.groupby(["stationName", "year", "week"])[col].agg(["mean", "std", "count"])
    weekly_stats = weekly_stats[weekly_stats["count"] >= 80]  # at least 80 hours per week
    weekly_stats["cv"] = weekly_stats["std"] / weekly_stats["mean"]
    weekly_stats = weekly_stats[weekly_stats["cv"].notna() & np.isfinite(weekly_stats["cv"])]
    station_weekly_cv = weekly_stats.groupby("stationName")["cv"].median()
    
    # CV vs measurement level (heteroscedasticity)
    # Bin by measurement level and compute CV per bin
    valid_hourly = valid.copy()
    valid_hourly["level_bin"] = pd.qcut(valid_hourly[col], q=10, duplicates="drop")
    level_cv = valid_hourly.groupby("level_bin", observed=True)[col].agg(["mean", "std"])
    level_cv["cv"] = level_cv["std"] / level_cv["mean"]
    
    noise_results[col] = {
        "daily_cv_median": float(station_cv.median()),
        "daily_cv_q25": float(station_cv.quantile(0.25)),
        "daily_cv_q75": float(station_cv.quantile(0.75)),
        "daily_cv_max": float(station_cv.quantile(0.95)),  # 95th percentile as "worst"
        "weekly_cv_median": float(station_weekly_cv.median()),
        "weekly_cv_q25": float(station_weekly_cv.quantile(0.25)),
        "weekly_cv_q75": float(station_weekly_cv.quantile(0.75)),
        "n_stations": int(station_cv.notna().sum()),
        "level_cv_data": {str(k): {"mean": float(v["mean"]), "cv": float(v["cv"])} 
                          for k, v in level_cv.iterrows() if np.isfinite(v["cv"])},
        "station_cv_values": station_cv.dropna().to_dict()
    }
    
    print(f"    Daily CV: median={station_cv.median():.3f}, IQR=[{station_cv.quantile(0.25):.3f}, {station_cv.quantile(0.75):.3f}]")
    print(f"    Weekly CV: median={station_weekly_cv.median():.3f}")

# Save noise results
with open(RESULTS_DIR / "noise_results.json", "w") as f:
    # Convert station_cv_values to simple format for JSON
    save_results = {}
    for col, res in noise_results.items():
        save_res = {k: v for k, v in res.items() if k != "station_cv_values"}
        save_results[col] = save_res
    json.dump(save_results, f, indent=2)

# Save full noise results with station-level data
with open(RESULTS_DIR / "noise_results_full.pkl", "wb") as f:
    pickle.dump(noise_results, f)

# ============================================================
# 2. MISSING DATA PATTERNS
# ============================================================
print("\n=== 2. Missing Data Patterns ===")

# Expected hours per station (roughly)
date_range = (df["dataTime"].max() - df["dataTime"].min()).total_seconds() / 3600
print(f"  Date range: {date_range:.0f} hours")

missing_results = {}

for col, label in SENSORS.items():
    station_missing = df.groupby("stationName")[col].apply(lambda x: x.isna().mean() * 100)
    
    # Flag rates
    flag_col = col.replace("Value", "Flag")
    if flag_col in df.columns:
        station_flagged = df.groupby("stationName")[flag_col].apply(
            lambda x: x.notna().mean() * 100  # flags indicate issues
        )
    else:
        station_flagged = pd.Series(0, index=station_missing.index)
    
    missing_results[col] = {
        "missing_pct_median": float(station_missing.median()),
        "missing_pct_mean": float(station_missing.mean()),
        "missing_pct_q95": float(station_missing.quantile(0.95)),
        "stations_over_50pct_missing": int((station_missing > 50).sum()),
        "flagged_pct_mean": float(station_flagged.mean()),
    }
    
    print(f"  {col}: median missing={station_missing.median():.1f}%, "
          f"stations>50% missing: {(station_missing > 50).sum()}")

# Data availability per station per day
print("\n  Computing daily availability matrix...")
df["date"] = df["dataTime"].dt.date
availability = df.groupby(["stationName", "date"]).size().unstack(fill_value=0)
availability_pct = (availability / 24 * 100).clip(upper=100)

# Overall availability by station type
avail_by_type = df.groupby("mangName").apply(
    lambda x: x[list(SENSORS.keys())].notna().mean() * 100
)
print(f"\n  Availability by station type:")
print(avail_by_type.to_string())

missing_results["availability_by_type"] = avail_by_type.to_dict()
missing_results["availability_matrix_shape"] = list(availability_pct.shape)

with open(RESULTS_DIR / "missing_results.json", "w") as f:
    json.dump(missing_results, f, indent=2, default=str)

# Save availability matrix
availability_pct.to_pickle(RESULTS_DIR / "availability_matrix.pkl")

# ============================================================
# 3. SENSOR DRIFT DETECTION
# ============================================================
print("\n=== 3. Sensor Drift Detection ===")

drift_results = {}

for col, label in SENSORS.items():
    print(f"\n  Processing {col}...")
    
    valid = df[df[col].notna()].copy()
    valid["date"] = valid["dataTime"].dt.date
    
    # Daily mean per station
    daily_mean = valid.groupby(["stationName", "date"])[col].mean().reset_index()
    daily_mean["date"] = pd.to_datetime(daily_mean["date"])
    
    # Regional daily mean (by sido + mangName)
    valid_with_meta = valid.copy()
    regional_daily = valid_with_meta.groupby(["sido", "mangName", "date"])[col].mean().reset_index()
    regional_daily = regional_daily.rename(columns={col: "regional_mean"})
    regional_daily["date"] = pd.to_datetime(regional_daily["date"])
    
    # Merge station metadata for sido/mangName
    station_meta = valid[["stationName", "sido", "mangName"]].drop_duplicates()
    daily_mean = daily_mean.merge(station_meta, on="stationName", how="left")
    daily_mean = daily_mean.merge(regional_daily, on=["sido", "mangName", "date"], how="left")
    
    # Compute deviation from regional mean
    daily_mean["deviation"] = daily_mean[col] - daily_mean["regional_mean"]
    daily_mean["day_num"] = (daily_mean["date"] - daily_mean["date"].min()).dt.days
    
    # For each station, fit linear regression to deviation over time
    drift_rates = {}
    drift_pvalues = {}
    
    stations_with_data = daily_mean.groupby("stationName").filter(
        lambda x: x["deviation"].notna().sum() >= 30  # at least 30 days
    )["stationName"].unique()
    
    for station in stations_with_data:
        sdata = daily_mean[daily_mean["stationName"] == station].dropna(subset=["deviation", "day_num"])
        if len(sdata) < 30:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            sdata["day_num"].values, sdata["deviation"].values
        )
        
        # Convert to unit/month
        drift_rate = slope * 30  # days to month
        drift_rates[station] = drift_rate
        drift_pvalues[station] = p_value
    
    drift_series = pd.Series(drift_rates)
    pvalue_series = pd.Series(drift_pvalues)
    
    # Significant drift = p < 0.05
    significant = pvalue_series[pvalue_series < 0.05]
    sig_drift = drift_series[significant.index]
    
    drift_results[col] = {
        "n_stations_analyzed": len(drift_rates),
        "n_significant_drift": len(significant),
        "pct_significant": float(len(significant) / max(len(drift_rates), 1) * 100),
        "drift_rate_median": float(drift_series.median()) if len(drift_series) > 0 else 0,
        "drift_rate_mean": float(drift_series.mean()) if len(drift_series) > 0 else 0,
        "drift_rate_std": float(drift_series.std()) if len(drift_series) > 0 else 0,
        "sig_drift_rate_median": float(sig_drift.median()) if len(sig_drift) > 0 else 0,
        "sig_drift_rate_abs_median": float(sig_drift.abs().median()) if len(sig_drift) > 0 else 0,
        "drift_rates": drift_series.to_dict(),
        "drift_pvalues": pvalue_series.to_dict(),
    }
    
    print(f"    Analyzed: {len(drift_rates)} stations")
    print(f"    Significant drift (p<0.05): {len(significant)} ({len(significant)/max(len(drift_rates),1)*100:.1f}%)")
    print(f"    Drift rate (sig): median={sig_drift.median():.4f} {SENSOR_UNITS[col]}/month" if len(sig_drift)>0 else "    No significant drift")

# Save drift results
with open(RESULTS_DIR / "drift_results.pkl", "wb") as f:
    pickle.dump(drift_results, f)

# Save summary (JSON-safe)
drift_summary = {}
for col, res in drift_results.items():
    drift_summary[col] = {k: v for k, v in res.items() if k not in ["drift_rates", "drift_pvalues"]}
with open(RESULTS_DIR / "drift_results.json", "w") as f:
    json.dump(drift_summary, f, indent=2)

# ============================================================
# 4. CROSS-SENSOR CORRELATION (PM2.5/PM10 ratio)
# ============================================================
print("\n=== 4. Cross-Sensor Correlation ===")

valid_pm = df[df["pm25Value"].notna() & df["pm10Value"].notna() & 
              (df["pm25Value"] > 0) & (df["pm10Value"] > 0)].copy()
valid_pm["pm_ratio"] = valid_pm["pm25Value"] / valid_pm["pm10Value"]

# Station-level ratio statistics
station_ratio = valid_pm.groupby("stationName")["pm_ratio"].agg(["mean", "std", "count"])
station_ratio = station_ratio[station_ratio["count"] >= 100]
station_ratio["cv"] = station_ratio["std"] / station_ratio["mean"]

# Weekly ratio stability per station
valid_pm["week"] = valid_pm["dataTime"].dt.isocalendar().week.to_numpy(dtype="int64", na_value=0)
valid_pm["year"] = valid_pm["dataTime"].dt.year
weekly_ratio = valid_pm.groupby(["stationName", "year", "week"])["pm_ratio"].mean().reset_index()
ratio_stability = weekly_ratio.groupby("stationName")["pm_ratio"].std()

cross_sensor_results = {
    "pm_ratio_overall_mean": float(valid_pm["pm_ratio"].mean()),
    "pm_ratio_overall_std": float(valid_pm["pm_ratio"].std()),
    "station_ratio_mean": float(station_ratio["mean"].mean()),
    "station_ratio_cv_median": float(station_ratio["cv"].median()),
    "ratio_temporal_stability_median": float(ratio_stability.median()),
    "n_stations": int(len(station_ratio)),
    "anomalous_ratio_stations": int(((station_ratio["mean"] < 0.3) | (station_ratio["mean"] > 0.9)).sum()),
}

print(f"  PM2.5/PM10 ratio: mean={cross_sensor_results['pm_ratio_overall_mean']:.3f}")
print(f"  Station ratio CV median: {cross_sensor_results['station_ratio_cv_median']:.3f}")
print(f"  Stations with anomalous ratio: {cross_sensor_results['anomalous_ratio_stations']}")

with open(RESULTS_DIR / "cross_sensor_results.json", "w") as f:
    json.dump(cross_sensor_results, f, indent=2)

# ============================================================
# 5. STATION FAILURE ANALYSIS
# ============================================================
print("\n=== 5. Station Failure Analysis ===")

failure_results = {}

# For PM2.5 as primary sensor
for col in ["pm25Value", "pm10Value"]:
    print(f"\n  Analyzing failures for {col}...")
    
    # Sort by station and time
    sorted_df = df[["stationName", "dataTime", col, "mangName"]].sort_values(
        ["stationName", "dataTime"]).copy()
    
    gap_counts = {"24h": 0, "72h": 0, "168h": 0}
    station_gaps = {}
    
    for station, group in sorted_df.groupby("stationName"):
        # Find consecutive missing periods
        missing = group[col].isna()
        if not missing.any():
            continue
        
        # Group consecutive missing periods
        missing_groups = (missing != missing.shift()).cumsum()
        missing_periods = group[missing].groupby(missing_groups)
        
        max_gap = 0
        n_gaps = 0
        for _, period in missing_periods:
            gap_hours = len(period)
            max_gap = max(max_gap, gap_hours)
            if gap_hours >= 24:
                gap_counts["24h"] += 1
                n_gaps += 1
            if gap_hours >= 72:
                gap_counts["72h"] += 1
            if gap_hours >= 168:
                gap_counts["168h"] += 1
        
        station_gaps[station] = {"max_gap": max_gap, "n_gaps_24h": n_gaps}
    
    n_stations = sorted_df["stationName"].nunique()
    stations_with_24h = sum(1 for v in station_gaps.values() if v["max_gap"] >= 24)
    stations_with_72h = sum(1 for v in station_gaps.values() if v["max_gap"] >= 72)
    stations_with_168h = sum(1 for v in station_gaps.values() if v["max_gap"] >= 168)
    
    failure_results[col] = {
        "total_stations": n_stations,
        "stations_with_gap_24h": stations_with_24h,
        "stations_with_gap_72h": stations_with_72h,
        "stations_with_gap_168h": stations_with_168h,
        "pct_24h": float(stations_with_24h / n_stations * 100),
        "pct_72h": float(stations_with_72h / n_stations * 100),
        "pct_168h": float(stations_with_168h / n_stations * 100),
        "total_gap_events_24h": gap_counts["24h"],
        "total_gap_events_72h": gap_counts["72h"],
        "total_gap_events_168h": gap_counts["168h"],
    }
    
    print(f"    Stations with >24h gap: {stations_with_24h} ({stations_with_24h/n_stations*100:.1f}%)")
    print(f"    Stations with >72h gap: {stations_with_72h} ({stations_with_72h/n_stations*100:.1f}%)")
    print(f"    Stations with >168h gap: {stations_with_168h} ({stations_with_168h/n_stations*100:.1f}%)")

# ============================================================
# 6. MTBF ESTIMATION
# ============================================================
print("\n=== 6. MTBF Estimation ===")

# MTBF = total operating hours / number of failures
# A "failure" = gap >= 24h
mtbf_by_type = {}

for mtype in df["mangName"].unique():
    type_df = df[df["mangName"] == mtype]
    n_stations = type_df["stationName"].nunique()
    
    # Total hours of data per station
    hours_per_station = type_df.groupby("stationName").size()
    total_hours = hours_per_station.sum()
    
    # Count failures (gaps >= 24h) for PM2.5
    n_failures = 0
    for station, group in type_df.sort_values("dataTime").groupby("stationName"):
        missing = group["pm25Value"].isna()
        if not missing.any():
            continue
        missing_groups = (missing != missing.shift()).cumsum()
        for _, period in group[missing].groupby(missing_groups):
            if len(period) >= 24:
                n_failures += 1
    
    mtbf = total_hours / max(n_failures, 1)
    mtbf_days = mtbf / 24
    
    mtbf_by_type[mtype] = {
        "n_stations": n_stations,
        "total_hours": int(total_hours),
        "n_failures": n_failures,
        "mtbf_hours": float(mtbf),
        "mtbf_days": float(mtbf_days),
    }
    
    print(f"  {mtype}: MTBF = {mtbf_days:.1f} days ({n_failures} failures across {n_stations} stations)")

failure_results["mtbf_by_type"] = mtbf_by_type

with open(RESULTS_DIR / "failure_results.json", "w") as f:
    json.dump(failure_results, f, indent=2, default=str)

print("\n=== All sensor characterization complete! ===")
print(f"Results saved to {RESULTS_DIR}")
