#!/usr/bin/env python3
"""
03_cross_calibration.py - Spatial cross-calibration and reference-based calibration.
"""
import os, json, pickle, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RESULTS_DIR = PROJECT / "results"

# Load data
print("Loading data...")
df = pd.read_pickle(DATA_DIR / "airquality_raw.pkl")
print(f"Shape: {df.shape}, Stations: {df['stationName'].nunique()}")

# Get station coordinates
station_coords = df.groupby("stationName")[["lat", "lon", "mangName", "sido"]].first()
station_coords = station_coords[station_coords["lat"].notna() & station_coords["lon"].notna()]
print(f"Stations with coords: {len(station_coords)}")

# ============================================================
# Haversine distance
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# 1. FIND ADJACENT STATION PAIRS (<5km)
# ============================================================
print("\n=== 1. Finding Adjacent Station Pairs ===")

pairs = []
station_names = station_coords.index.tolist()

for i in range(len(station_names)):
    for j in range(i+1, len(station_names)):
        s1, s2 = station_names[i], station_names[j]
        d = haversine(
            station_coords.loc[s1, "lat"], station_coords.loc[s1, "lon"],
            station_coords.loc[s2, "lat"], station_coords.loc[s2, "lon"]
        )
        if d < 5.0:
            pairs.append((s1, s2, d))

print(f"Found {len(pairs)} station pairs within 5km")

# Also find pairs within 10km for more coverage
pairs_10km = []
for i in range(len(station_names)):
    for j in range(i+1, len(station_names)):
        s1, s2 = station_names[i], station_names[j]
        d = haversine(
            station_coords.loc[s1, "lat"], station_coords.loc[s1, "lon"],
            station_coords.loc[s2, "lat"], station_coords.loc[s2, "lon"]
        )
        if d < 10.0:
            pairs_10km.append((s1, s2, d))

print(f"Found {len(pairs_10km)} station pairs within 10km")

# ============================================================
# 2. SPATIAL CROSS-CALIBRATION
# ============================================================
print("\n=== 2. Spatial Cross-Calibration ===")

# Create hourly pivot for PM2.5
pm25_hourly = df.pivot_table(
    index="dataTime", columns="stationName", values="pm25Value", aggfunc="first"
)

cross_cal_results = []
use_pairs = pairs if len(pairs) >= 50 else pairs_10km
pair_label = "5km" if len(pairs) >= 50 else "10km"
print(f"Using {len(use_pairs)} pairs within {pair_label}")

for s1, s2, dist in use_pairs:
    if s1 not in pm25_hourly.columns or s2 not in pm25_hourly.columns:
        continue
    
    common = pm25_hourly[[s1, s2]].dropna()
    if len(common) < 100:
        continue
    
    corr = common[s1].corr(common[s2])
    bias = (common[s1] - common[s2]).mean()
    bias_std = (common[s1] - common[s2]).std()
    rmse = np.sqrt(((common[s1] - common[s2])**2).mean())
    
    # Types
    t1 = station_coords.loc[s1, "mangName"] if s1 in station_coords.index else ""
    t2 = station_coords.loc[s2, "mangName"] if s2 in station_coords.index else ""
    
    cross_cal_results.append({
        "station1": s1,
        "station2": s2,
        "distance_km": float(dist),
        "type1": t1,
        "type2": t2,
        "n_common": int(len(common)),
        "correlation": float(corr),
        "bias": float(bias),
        "bias_std": float(bias_std),
        "rmse": float(rmse),
        "needs_calibration": abs(bias) > 5.0,  # threshold: 5 μg/m³
    })

print(f"Analyzed {len(cross_cal_results)} pairs")

if cross_cal_results:
    cal_df = pd.DataFrame(cross_cal_results)
    print(f"\nCorrelation: median={cal_df['correlation'].median():.3f}, "
          f"min={cal_df['correlation'].min():.3f}, max={cal_df['correlation'].max():.3f}")
    print(f"Bias: median={cal_df['bias'].median():.2f}, "
          f"mean abs={cal_df['bias'].abs().mean():.2f} μg/m³")
    print(f"RMSE: median={cal_df['rmse'].median():.2f} μg/m³")
    print(f"Pairs needing calibration (|bias|>5): {cal_df['needs_calibration'].sum()} "
          f"({cal_df['needs_calibration'].mean()*100:.1f}%)")

# ============================================================
# 3. REFERENCE-BASED CALIBRATION
# ============================================================
print("\n=== 3. Reference-Based Calibration ===")

# 국가배경농도(도서) stations as reference
ref_stations = station_coords[station_coords["mangName"] == "국가배경농도(도서)"].index.tolist()
urban_stations = station_coords[station_coords["mangName"] == "도시대기"].index.tolist()

print(f"Reference stations: {len(ref_stations)}")
print(f"Urban stations: {len(urban_stations)}")

ref_cal_results = []

for ref in ref_stations:
    if ref not in pm25_hourly.columns:
        continue
    
    ref_lat = station_coords.loc[ref, "lat"]
    ref_lon = station_coords.loc[ref, "lon"]
    
    # Find nearest urban stations (within 100km for island stations)
    for urban in urban_stations:
        if urban not in pm25_hourly.columns:
            continue
        
        dist = haversine(
            ref_lat, ref_lon,
            station_coords.loc[urban, "lat"], station_coords.loc[urban, "lon"]
        )
        
        if dist > 100:  # within 100km
            continue
        
        common = pm25_hourly[[ref, urban]].dropna()
        if len(common) < 100:
            continue
        
        corr = common[ref].corr(common[urban])
        bias = (common[urban] - common[ref]).mean()
        
        ref_cal_results.append({
            "reference": ref,
            "urban": urban,
            "distance_km": float(dist),
            "n_common": int(len(common)),
            "correlation": float(corr),
            "bias_urban_minus_ref": float(bias),
        })

print(f"Reference-urban pairs analyzed: {len(ref_cal_results)}")
if ref_cal_results:
    ref_df = pd.DataFrame(ref_cal_results)
    print(f"  Correlation: median={ref_df['correlation'].median():.3f}")
    print(f"  Bias (urban-ref): median={ref_df['bias_urban_minus_ref'].median():.2f} μg/m³")

# ============================================================
# 4. TEMPORAL STABILITY OF CALIBRATION BIAS
# ============================================================
print("\n=== 4. Temporal Stability of Bias ===")

temporal_stability = []

# Use top pairs with highest correlation
if cross_cal_results:
    cal_df_sorted = cal_df.sort_values("correlation", ascending=False)
    top_pairs = cal_df_sorted.head(min(50, len(cal_df_sorted)))
    
    for _, row in top_pairs.iterrows():
        s1, s2 = row["station1"], row["station2"]
        
        if s1 not in pm25_hourly.columns or s2 not in pm25_hourly.columns:
            continue
        
        common = pm25_hourly[[s1, s2]].dropna().copy()
        if len(common) < 200:
            continue
        
        common["bias"] = common[s1] - common[s2]
        common["date"] = common.index.date
        
        # Weekly bias
        common["week"] = common.index.isocalendar().week.to_numpy(dtype="int64", na_value=0)
        common["year"] = common.index.year
        weekly_bias = common.groupby(["year", "week"])["bias"].mean()
        
        if len(weekly_bias) < 4:
            continue
        
        # Trend in bias over time
        x = np.arange(len(weekly_bias))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, weekly_bias.values)
        
        temporal_stability.append({
            "station1": s1,
            "station2": s2,
            "bias_trend_slope": float(slope),
            "bias_trend_pvalue": float(p_value),
            "bias_std_weekly": float(weekly_bias.std()),
            "bias_range": float(weekly_bias.max() - weekly_bias.min()),
            "n_weeks": int(len(weekly_bias)),
            "is_drifting": p_value < 0.05,
        })

print(f"Temporal stability analyzed for {len(temporal_stability)} pairs")
if temporal_stability:
    temp_df = pd.DataFrame(temporal_stability)
    print(f"  Pairs with drifting bias (p<0.05): {temp_df['is_drifting'].sum()} "
          f"({temp_df['is_drifting'].mean()*100:.1f}%)")
    print(f"  Weekly bias std: median={temp_df['bias_std_weekly'].median():.2f} μg/m³")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
all_cross_cal = {
    "spatial_pairs": cross_cal_results,
    "reference_calibration": ref_cal_results,
    "temporal_stability": temporal_stability,
    "summary": {
        "n_pairs_analyzed": len(cross_cal_results),
        "pair_distance_threshold": pair_label,
        "median_correlation": float(cal_df["correlation"].median()) if cross_cal_results else 0,
        "median_abs_bias": float(cal_df["bias"].abs().median()) if cross_cal_results else 0,
        "median_rmse": float(cal_df["rmse"].median()) if cross_cal_results else 0,
        "pct_needing_calibration": float(cal_df["needs_calibration"].mean()*100) if cross_cal_results else 0,
        "n_ref_pairs": len(ref_cal_results),
        "n_temporal_pairs": len(temporal_stability),
        "pct_drifting_bias": float(temp_df["is_drifting"].mean()*100) if temporal_stability else 0,
    }
}

with open(RESULTS_DIR / "cross_calibration_results.pkl", "wb") as f:
    pickle.dump(all_cross_cal, f)

with open(RESULTS_DIR / "cross_calibration_summary.json", "w") as f:
    json.dump(all_cross_cal["summary"], f, indent=2)

print("\n=== Cross-calibration analysis complete! ===")
