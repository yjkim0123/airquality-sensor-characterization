#!/usr/bin/env python3
"""Collect remaining stations - single thread, generous rate limiting."""
import os, json, time, requests, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RAW_DIR = DATA_DIR / "raw_stations"

API_KEY = "41de1ce8903be5cd834094374a879625836ac3743c641b7886f3628ac3c697ef"
URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

with open(DATA_DIR / "stations.json") as f:
    stations = json.load(f)

to_collect = [s for s in stations if not (RAW_DIR / f"{s['stationName']}.json").exists()]
print(f"Remaining: {len(to_collect)} stations", flush=True)

errors = []
collected = 0
consecutive_429 = 0

for i, st in enumerate(to_collect):
    name = st["stationName"]
    outfile = RAW_DIR / f"{name}.json"
    
    success = False
    for attempt in range(3):
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
            
            if r.status_code == 429:
                consecutive_429 += 1
                wait = min(10 * consecutive_429, 60)
                print(f"  429 for {name}, waiting {wait}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
                continue
            
            consecutive_429 = 0
            r.raise_for_status()
            data = r.json()
            body = data.get("response", {}).get("body", {})
            items = body.get("items", [])
            
            if items and body.get("totalCount", 0) > 0:
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False)
                collected += 1
                success = True
            else:
                errors.append({"station": name, "error": "no data"})
                success = True  # don't retry no-data
            break
        except Exception as e:
            if attempt == 2:
                errors.append({"station": name, "error": str(e)})
            else:
                time.sleep(3)
    
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(to_collect)}] Collected: {collected} | Errors: {len(errors)}", flush=True)
    
    time.sleep(1.0)  # more conservative

print(f"\nDone: {collected} new, {len(errors)} errors", flush=True)
with open(DATA_DIR / "collection_errors2.json", "w") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)
