#!/usr/bin/env python3
"""
04_generate_figures.py - Generate 4 figures for IEEE Sensors Letters.
"""
import os, json, pickle, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RESULTS_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# Load data
print("Loading data...")
df = pd.read_pickle(DATA_DIR / "airquality_raw.pkl")

with open(DATA_DIR / "stations.json") as f:
    stations = json.load(f)
stations_meta = {s["stationName"]: s for s in stations}

with open(RESULTS_DIR / "noise_results_full.pkl", "rb") as f:
    noise_results = pickle.load(f)

with open(RESULTS_DIR / "drift_results.pkl", "rb") as f:
    drift_results = pickle.load(f)

with open(RESULTS_DIR / "cross_calibration_results.pkl", "rb") as f:
    cross_cal = pickle.load(f)

availability_matrix = pd.read_pickle(RESULTS_DIR / "availability_matrix.pkl")

SENSORS = {
    "pm25Value": r"PM$_{2.5}$",
    "pm10Value": r"PM$_{10}$",
    "so2Value": r"SO$_2$",
    "no2Value": r"NO$_2$",
    "coValue": "CO",
    "o3Value": r"O$_3$"
}

TYPE_COLORS = {
    "도시대기": "#1f77b4",
    "도로변대기": "#ff7f0e",
    "교외대기": "#2ca02c",
    "국가배경농도(도서)": "#d62728",
    "항만": "#9467bd",
}

TYPE_LABELS = {
    "도시대기": "Urban",
    "도로변대기": "Roadside",
    "교외대기": "Suburban",
    "국가배경농도(도서)": "Background",
    "항만": "Port",
}

# ============================================================
# FIGURE 1: (a) Station map colored by type, (b) Data availability heatmap
# ============================================================
print("Generating Figure 1...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

# (a) Station map
station_df = pd.DataFrame(stations)
station_df["lat"] = pd.to_numeric(station_df["dmX"], errors="coerce")
station_df["lon"] = pd.to_numeric(station_df["dmY"], errors="coerce")
station_df = station_df[station_df["lat"].notna() & station_df["lon"].notna()]

# Only plot stations we have data for
collected_stations = set(df["stationName"].unique())
station_df["collected"] = station_df["stationName"].isin(collected_stations)

for mtype, color in TYPE_COLORS.items():
    subset = station_df[station_df["mangName"] == mtype]
    collected = subset[subset["collected"]]
    ax1.scatter(collected["lon"], collected["lat"], c=color, s=8, alpha=0.7,
                label=f"{TYPE_LABELS[mtype]} ({len(collected)})", zorder=3, edgecolors="none")

ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")
ax1.set_title("(a) Station Distribution")
ax1.legend(loc="lower left", fontsize=5.5, framealpha=0.9, markerscale=1.5)
ax1.set_xlim(124.5, 131.5)
ax1.set_ylim(33, 39)
ax1.set_aspect("equal")

# (b) Bar chart: mean % missing data by pollutant (from missing_results.json)
import json
with open(RESULTS_DIR / "missing_results.json") as _f:
    missing_data = json.load(_f)

pollutant_keys = ["pm25Value", "pm10Value", "so2Value", "no2Value", "coValue", "o3Value"]
poll_labels    = [r"PM$_{2.5}$", r"PM$_{10}$", r"SO$_2$", r"NO$_2$", "CO", r"O$_3$"]
miss_pcts  = [missing_data[k]["missing_pct_mean"]       for k in pollutant_keys if k in missing_data]
miss_n50   = [missing_data[k]["stations_over_50pct_missing"] for k in pollutant_keys if k in missing_data]
miss_q95   = [missing_data[k]["missing_pct_q95"]        for k in pollutant_keys if k in missing_data]
valid_labels = [poll_labels[i] for i, k in enumerate(pollutant_keys) if k in missing_data]

bar_colors = plt.cm.Set2(np.linspace(0, 1, len(valid_labels)))
bars = ax2.bar(valid_labels, miss_pcts, color=bar_colors, edgecolor="white", linewidth=0.5)

# Add 95th-percentile error cap
for bar, q95, n50 in zip(bars, miss_q95, miss_n50):
    xc = bar.get_x() + bar.get_width() / 2
    ax2.plot([xc - 0.15, xc + 0.15], [q95, q95], color="gray", linewidth=1.0)
    ax2.text(xc, q95 + 0.1, f"n>{50}%: {n50}",
             ha="center", va="bottom", fontsize=5.5, color="#374151")

ax2.set_ylabel("Mean Missing Data (%)")
ax2.set_title("(b) Data Missingness by Pollutant\n(cap = 95th pct; labels = stations >50% missing)")
ax2.set_ylim(0, max(miss_q95) * 1.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_stations_availability.pdf")
fig.savefig(FIG_DIR / "fig1_stations_availability.png")
print("  Figure 1 saved.")

# ============================================================
# FIGURE 2: (a) Noise CV distribution by pollutant, (b) CV vs measurement level
# ============================================================
print("Generating Figure 2...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

# (a) CV distribution boxplot
cv_data = []
labels = []
for col, label in SENSORS.items():
    if col in noise_results:
        vals = list(noise_results[col]["station_cv_values"].values())
        cv_data.append(vals)
        labels.append(label)

bp = ax1.boxplot(cv_data, labels=labels, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.5),
                 whiskerprops=dict(linewidth=0.8),
                 boxprops=dict(linewidth=0.8))

colors = plt.cm.Set2(np.linspace(0, 1, len(cv_data)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel("Daily Coefficient of Variation")
ax1.set_title("(a) Sensor Noise by Pollutant")
ax1.set_ylim(0, 1.0)

# (b) CV vs measurement level for PM2.5 and PM10 — with std error bars
for col, label, marker, color in [
    ("pm25Value", r"PM$_{2.5}$", "o", "#1f77b4"),
    ("pm10Value", r"PM$_{10}$", "s", "#ff7f0e"),
]:
    if col in noise_results and "level_cv_data" in noise_results[col]:
        level_data = noise_results[col]["level_cv_data"]
        means = [v["mean"] for v in level_data.values()]
        cvs   = [v["cv"]   for v in level_data.values()]
        stds  = [v.get("std", 0.0) for v in level_data.values()]
        ax2.errorbar(means, cvs, yerr=stds, marker=marker, ms=5,
                     color=color, label=label, alpha=0.85,
                     linestyle="-", linewidth=0.8, capsize=3, capthick=0.8)

ax2.set_xlabel(r"Concentration Level ($\mu$g m$^{-3}$)")
ax2.set_ylabel("Coefficient of Variation")
ax2.set_title("(b) Heteroscedasticity\n(error bars = ±1 SD across stations)")
ax2.legend(fontsize=7)
ax2.set_ylim(0, None)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_noise_analysis.pdf")
fig.savefig(FIG_DIR / "fig2_noise_analysis.png")
print("  Figure 2 saved.")

# ============================================================
# FIGURE 3: (a) Drift example, (b) Drift rate distribution
# ============================================================
print("Generating Figure 3...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

# (a) Drift example — find station with most significant drift in PM2.5
col = "pm25Value"
drift_rates = drift_results[col]["drift_rates"]
drift_pvals = drift_results[col]["drift_pvalues"]

# Find station with significant drift and largest absolute rate
sig_stations = {s: abs(drift_rates[s]) for s in drift_pvals if drift_pvals[s] < 0.01}
if sig_stations:
    example_station = max(sig_stations, key=sig_stations.get)
else:
    example_station = max(drift_rates, key=lambda x: abs(drift_rates[x]))

# Plot deviation from regional mean for example station
valid = df[df[col].notna() & (df["stationName"] == example_station)].copy()
valid["date"] = valid["dataTime"].dt.date
daily_mean = valid.groupby("date")[col].mean()

# Regional mean
sido = df[df["stationName"] == example_station]["sido"].iloc[0]
mangName = df[df["stationName"] == example_station]["mangName"].iloc[0]
regional = df[(df["sido"] == sido) & (df["mangName"] == mangName) & df[col].notna()].copy()
regional["date"] = regional["dataTime"].dt.date
regional_daily = regional.groupby("date")[col].mean()

common_dates = daily_mean.index.intersection(regional_daily.index)
deviation = daily_mean[common_dates] - regional_daily[common_dates]

# 7-day rolling
dev_series = pd.Series(deviation.values, index=pd.to_datetime(list(common_dates)))
rolling_dev = dev_series.rolling(7, min_periods=3).mean()

ax1.plot(rolling_dev.index, rolling_dev.values, "b-", linewidth=0.8, alpha=0.8)
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

# Add trend line
x_num = np.arange(len(rolling_dev.dropna()))
y_vals = rolling_dev.dropna().values
if len(x_num) > 10:
    slope, intercept, _, _, _ = stats.linregress(x_num, y_vals)
    ax1.plot(rolling_dev.dropna().index, intercept + slope * x_num, "r--", linewidth=1.0,
             label=f"Trend: {slope*30:.2f}/month")
    ax1.legend(fontsize=6)

ax1.set_ylabel(r"Deviation from Regional Mean ($\mu$g m$^{-3}$)")
ax1.set_title("(a) Representative PM$_{2.5}$ Drift Example")
ax1.tick_params(axis="x", rotation=30)
# Format x-axis dates
ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))

# (b) Drift rate distribution
for idx, (col, label) in enumerate(list(SENSORS.items())[:3]):
    if col in drift_results:
        rates = list(drift_results[col]["drift_rates"].values())
        pvals = list(drift_results[col]["drift_pvalues"].values())
        sig_rates = [r for r, p in zip(rates, pvals) if p < 0.05]
        
        color = plt.cm.Set1(idx / 6)
        ax2.hist(sig_rates, bins=30, alpha=0.5, color=color, label=label, density=True)

ax2.set_xlabel(r"Drift Rate (normalised, $\sigma$/month)")
ax2.set_ylabel("Density")
ax2.set_title("(b) Significant Drift Rate Distribution\n"
              r"(PM$_{2.5}$/$_{10}$: $\mu$g m$^{-3}$/mo; SO$_2$: $\times10^{-3}$ ppm/mo)")
ax2.legend(fontsize=6)
ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_drift_analysis.pdf")
fig.savefig(FIG_DIR / "fig3_drift_analysis.png")
print("  Figure 3 saved.")

# ============================================================
# FIGURE 4: (a) Adjacent station scatter, (b) Bias stability
# ============================================================
print("Generating Figure 4...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

# (a) Adjacent station scatter plot — pick the pair with highest correlation
spatial_pairs = cross_cal["spatial_pairs"]
if spatial_pairs:
    sorted_pairs = sorted(spatial_pairs, key=lambda x: x["correlation"], reverse=True)
    best = sorted_pairs[0]
    s1, s2 = best["station1"], best["station2"]
    
    pm25_hourly = df.pivot_table(index="dataTime", columns="stationName", values="pm25Value", aggfunc="first")
    
    if s1 in pm25_hourly.columns and s2 in pm25_hourly.columns:
        common = pm25_hourly[[s1, s2]].dropna()
        ax1.scatter(common[s1], common[s2], s=1, alpha=0.15, c="#1f77b4")
        
        # 1:1 line
        max_val = max(common[s1].max(), common[s2].max())
        ax1.plot([0, max_val], [0, max_val], "r--", linewidth=0.8, label="1:1 line")
        
        # Regression line
        slope, intercept, r, _, _ = stats.linregress(common[s1], common[s2])
        x_range = np.linspace(0, max_val, 100)
        ax1.plot(x_range, intercept + slope * x_range, "k-", linewidth=0.8,
                 label=f"r={r:.3f}, d={best['distance_km']:.1f}km")
        
        ax1.set_xlabel(r"Station A PM$_{2.5}$ ($\mu$g m$^{-3}$)")
        ax1.set_ylabel(r"Station B PM$_{2.5}$ ($\mu$g m$^{-3}$)")
        ax1.set_title(f"(a) Adjacent Station Correlation\n"
                      f"(distance = {best['distance_km']:.1f} km, "
                      f"r = {best['correlation']:.3f})")
        ax1.legend(fontsize=6, loc="upper left")
        ax1.set_xlim(0, None)
        ax1.set_ylim(0, None)

# (b) Correlation vs distance + Bias stability
if spatial_pairs:
    cal_df = pd.DataFrame(spatial_pairs)
    
    # Scatter of correlation vs distance
    ax2.scatter(cal_df["distance_km"], cal_df["correlation"], s=5, alpha=0.4, c="#1f77b4")
    
    # Binned trend
    cal_df["dist_bin"] = pd.cut(cal_df["distance_km"], bins=10)
    binned = cal_df.groupby("dist_bin", observed=True)["correlation"].agg(["mean", "std", "count"])
    bin_centers = [(b.left + b.right) / 2 for b in binned.index]
    ax2.plot(bin_centers, binned["mean"], "ro-", markersize=4, linewidth=1.2, label="Binned mean")
    
    ax2.set_xlabel("Inter-station Distance (km)")
    ax2.set_ylabel(r"PM$_{2.5}$ Pearson Correlation")
    ax2.set_title("(b) Spatial Correlation Decay\n(279 pairs within 5 km)")
    ax2.legend(fontsize=6)
    ax2.set_ylim(max(0.0, cal_df["correlation"].min() - 0.05), 1.0)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_cross_calibration.pdf")
fig.savefig(FIG_DIR / "fig4_cross_calibration.png")
print("  Figure 4 saved.")

print("\n=== All figures generated! ===")
print(f"Saved to {FIG_DIR}")
