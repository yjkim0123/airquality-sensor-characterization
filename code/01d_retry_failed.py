#!/usr/bin/env python3
"""Retry failed stations with slower rate limiting."""
import os, json, time, requests, sys, signal
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT = Path("/Users/yongjun_kim/Documents/project_airquality_sensor")
DATA_DIR = PROJECT / "data"
RAW_DIR = DATA_DIR / "raw_stations"

API_KEY = "41de1ce8903be5cd834094374a879625836ac3743c641b7886f3628ac3c697ef"
URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

# Read error list - only retry 429 errors
with open(DATA_DIR / "collection_errors.json") as f:
    errors = json.load(f)

to_retry = [e["station"] for e in errors if "429" in e.get("error", "")]
# Filter out already collected
to_retry = [s for s in to_retry if not (RAW_DIR / f"{s}.json").exists()]
print(f"Stations to retry: {len(to_retry)}", flush=True)

collected = 0
new_errors = []
delay = 1.5
consecutive_429 = 0
start_time = time.time()

for i, name in enumerate(to_retry):
    # Check 5 minute timeout
    elapsed = time.time() - start_time
    if elapsed > 300:
        print(f"\n5 minute timeout reached after {i} stations. Stopping.", flush=True)
        break
    
    outfile = RAW_DIR / f"{name}.json"
    if outfile.exists():
        continue
    
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
            if consecutive_429 >= 3:
                delay = 3.0
                print(f"  Increasing delay to 3s after {consecutive_429} consecutive 429s", flush=True)
            if consecutive_429 >= 10:
                print(f"  Too many 429s ({consecutive_429}). Stopping.", flush=True)
                new_errors.append({"station": name, "error": "429 - stopped"})
                # Add remaining stations as errors
                for remaining in to_retry[i+1:]:
                    if not (RAW_DIR / f"{remaining}.json").exists():
                        new_errors.append({"station": remaining, "error": "429 - not attempted"})
                break
            new_errors.append({"station": name, "error": f"429"})
            time.sleep(delay * 2)
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
        else:
            new_errors.append({"station": name, "error": "no data"})
        
    except Exception as e:
        new_errors.append({"station": name, "error": str(e)})
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(to_retry)}] Collected: {collected} | Errors: {len(new_errors)} | {elapsed:.0f}s elapsed", flush=True)
    
    time.sleep(delay)

elapsed = time.time() - start_time
total_files = len(list(RAW_DIR.glob("*.json")))
print(f"\nDone in {elapsed:.0f}s: {collected} new stations collected, {len(new_errors)} errors", flush=True)
print(f"Total station files now: {total_files}", flush=True)

with open(DATA_DIR / "collection_errors_retry.json", "w") as f:
    json.dump(new_errors, f, ensure_ascii=False, indent=2)
