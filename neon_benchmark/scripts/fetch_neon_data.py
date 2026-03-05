"""
NEON Lake Water Temperature Benchmark: Data Acquisition v2

Downloads four specific NEON data products for a lake site:
  1. DP1.20264.001  Water temperature (thermistor, shallowest depth)
  2. DP1.00024.001  PAR (photosynthetically active radiation)
  3. DP1.20271.001  RH and air temperature above water on-buoy (tempRHMean)
  4. DP1.00001.001  2D wind speed

All four must provide REAL data -- no synthetic fallbacks for air temp
or wind. If the primary site is missing buoy air temp, the script
automatically tries fallback sites (SUGG, TOOL, BARC).

Output: single CSV with columns:
    timestamp, time_delta, water_temp, air_temp, wind_speed, par

Usage:
    python -u neon_benchmark/scripts/fetch_neon_data.py

Non-destructive: all outputs go to neon_benchmark/data/
"""

import os, sys, json, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
CONFIG_PATH = BENCH_DIR / "configs" / "pipeline_config.json"
RAW_DIR = BENCH_DIR / "data" / "raw"
PROCESSED_DIR = BENCH_DIR / "data" / "processed"
SPLITS_DIR = BENCH_DIR / "data" / "splits"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, 'r') as f:
    CFG = json.load(f)

YEAR_MONTHS = CFG['year_months']
DT = CFG['dt_seconds']
TRAIN_FRAC = CFG['train_frac']
VAL_FRAC = CFG['val_frac']
TEST_FRAC = CFG['test_frac']

NEON_API_BASE = "https://data.neonscience.org/api/v0"

# Four specific NEON products
PRODUCTS = {
    'water_temp': 'DP1.20264.001',
    'par':        'DP1.00024.001',
    'buoy_air':   'DP1.20271.001',
    'wind':       'DP1.00001.001',
}

# Site priority: try primary first, then fallbacks
SITE_PRIORITY = [CFG['site_code'], 'SUGG', 'TOOL', 'BARC', 'CRAM', 'LIRO']
# De-duplicate while preserving order
SITE_PRIORITY = list(dict.fromkeys(SITE_PRIORITY))


# ==============================================================================
#  NEON API HELPERS
# ==============================================================================

def get_data_urls(product_code, site_code, year_month):
    """Query NEON API for data file URLs."""
    url = f"{NEON_API_BASE}/data/{product_code}/{site_code}/{year_month}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        if 'data' not in data or 'files' not in data['data']:
            return []
        return data['data']['files']
    except requests.exceptions.RequestException as e:
        print(f"    API error: {e}")
        return []


def find_file(files, patterns, prefer_30min=True, return_all=False):
    """Find CSV file(s) matching patterns. Prefer 30min and basic.
    If return_all=True, return list of ALL matching files (for multi-sensor products).
    """
    csv_files = [f for f in files if f['name'].endswith('.csv')]
    if not csv_files:
        return [] if return_all else None

    for pattern in patterns:
        matches = [f for f in csv_files if pattern in f['name']]
        if matches:
            if prefer_30min:
                m30 = [f for f in matches if '30min' in f['name']
                       or '030' in f['name']
                       or '30_min' in f['name']]
                if m30:
                    matches = m30
            # Prefer 'basic' over 'expanded' (smaller files)
            basic = [f for f in matches if 'basic' in f['name']]
            if basic:
                matches = basic

            if return_all:
                return matches
            return matches[0]
    return [] if return_all else None


def download_csv(file_url, save_path, force=False):
    """Download a CSV file from NEON, with caching."""
    if save_path.exists() and not force:
        print(f"    Cached: {save_path.name}")
        return pd.read_csv(save_path)

    print(f"    Downloading: {save_path.name}")
    resp = requests.get(file_url, timeout=120)
    resp.raise_for_status()
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    return pd.read_csv(StringIO(resp.text))


# ==============================================================================
#  PRODUCT-SPECIFIC DOWNLOADERS
# ==============================================================================

def download_product(product_code, site, year_months, label,
                     file_patterns, force=False, multi_sensor=False):
    """Download all months of a product, return concatenated DataFrame.
    If multi_sensor=True, download ALL matching files per month (e.g. multiple depths).
    """
    print(f"\n--- Downloading {label} ({product_code}) @ {site} ---")
    all_dfs = []

    for ym in year_months:
        print(f"  {ym}:", end="")
        files = get_data_urls(product_code, site, ym)
        if not files:
            print(f" no data")
            continue

        if multi_sensor:
            targets = find_file(files, file_patterns, return_all=True)
            if not targets:
                print(f" no matching files (tried {file_patterns})")
                continue
            print(f" {len(targets)} sensor files")
            for i, tgt in enumerate(targets):
                save_name = f"{label}_{site}_{ym}_{i}.csv"
                save_path = RAW_DIR / save_name
                df = download_csv(tgt['url'], save_path, force=force)
                all_dfs.append(df)
        else:
            target = find_file(files, file_patterns)
            if target is None:
                print(f" no matching file (tried {file_patterns})")
                continue
            save_name = f"{label}_{site}_{ym}.csv"
            save_path = RAW_DIR / save_name
            df = download_csv(target['url'], save_path, force=force)
            all_dfs.append(df)

    if not all_dfs:
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total {label} rows: {len(df_all)}")
    print(f"  Columns: {list(df_all.columns)[:8]}...")
    return df_all


def check_product_availability(product_code, site, year_months, label):
    """Quick check: does this product have data for at least 1 month?"""
    for ym in year_months:
        files = get_data_urls(product_code, site, ym)
        if files:
            return True
    return False


# ==============================================================================
#  SITE SELECTION (auto-fallback)
# ==============================================================================

def select_site():
    """Find a site with all four products available."""
    print("\n--- Site Selection (checking product availability) ---")

    for site in SITE_PRIORITY:
        print(f"\n  Checking {site}...")
        available = {}
        for key, prod_code in PRODUCTS.items():
            ok = check_product_availability(prod_code, site,
                                            YEAR_MONTHS, key)
            status = "OK" if ok else "MISSING"
            print(f"    {key} ({prod_code}): {status}")
            available[key] = ok

        if all(available.values()):
            print(f"\n  >>> Selected site: {site} (all 4 products available)")
            return site

        missing = [k for k, v in available.items() if not v]
        print(f"    Missing: {missing}")

    raise RuntimeError(
        f"No site has all 4 products for {YEAR_MONTHS}. "
        f"Tried: {SITE_PRIORITY}"
    )


# ==============================================================================
#  PARSE EACH PRODUCT
# ==============================================================================

def parse_water_temp(df):
    """Parse DP1.20264.001 -> (datetime, water_temp) at shallowest depth."""
    # Column: tsdWaterTemp (1-min) or surfWaterTempMean (30-min)
    time_col = 'startDateTime'
    temp_col = None
    for c in ['tsdWaterTempMean', 'tsdWaterTemp', 'surfWaterTempMean', 'surfWaterTemp']:
        if c in df.columns:
            temp_col = c
            break
    if temp_col is None:
        for c in df.columns:
            if 'temp' in c.lower() and 'expuncert' not in c.lower() \
               and 'finalqf' not in c.lower() and 'end' not in c.lower():
                temp_col = c
                break
    if temp_col is None:
        raise ValueError(f"No temp column in water_temp. Cols: {list(df.columns)}")

    print(f"  Water temp column: {temp_col}")

    # Filter to shallowest depth
    if 'thermistorDepth' in df.columns:
        depths = df['thermistorDepth'].dropna().unique()
        if len(depths) > 0:
            shallowest = min(depths)
            df = df[df['thermistorDepth'] == shallowest].copy()
            print(f"  Filtered to depth={shallowest}m -> {len(df)} rows")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    out = df[[time_col, temp_col]].copy()
    out.columns = ['datetime', 'water_temp']
    out = out.dropna(subset=['water_temp'])
    out = out.set_index('datetime').resample('30min').mean().dropna()
    return out.reset_index()


def parse_par(df):
    """Parse DP1.00024.001 -> (datetime, par)."""
    time_col = 'startDateTime'
    par_col = None
    for c in ['PARMean', 'outPARMean']:
        if c in df.columns:
            par_col = c
            break
    if par_col is None:
        raise ValueError(f"No PAR column. Cols: {list(df.columns)}")

    print(f"  PAR column: {par_col}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    out = df[[time_col, par_col]].copy()
    out.columns = ['datetime', 'par']
    out = out.dropna(subset=['par'])
    out['par'] = out['par'].clip(lower=0)
    out = out.set_index('datetime').resample('30min').mean().dropna()
    return out.reset_index()


def parse_buoy_air_temp(df):
    """Parse DP1.20271.001 -> (datetime, air_temp).
    Column is tempRHMean (temp from RH sensor on buoy)."""
    time_col = 'startDateTime'
    air_col = None
    for c in ['tempRHMean', 'tempSingleMean', 'buoyTempMean', 'airTempMean']:
        if c in df.columns:
            air_col = c
            break
    if air_col is None:
        # Broader search for temp*Mean columns (exclude QF, Uncert, etc.)
        for c in df.columns:
            cl = c.lower()
            if 'temp' in cl and 'mean' in cl \
               and 'expuncert' not in cl and 'finalqf' not in cl \
               and 'end' not in cl and 'water' not in cl \
               and 'dew' not in cl:
                air_col = c
                break
    if air_col is None:
        raise ValueError(f"No air temp column in buoy data. Cols: {list(df.columns)}")

    print(f"  Buoy air temp column: {air_col}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    out = df[[time_col, air_col]].copy()
    out.columns = ['datetime', 'air_temp']
    out = out.dropna(subset=['air_temp'])
    out = out.set_index('datetime').resample('30min').mean().dropna()
    return out.reset_index()


def parse_wind(df):
    """Parse DP1.00001.001 -> (datetime, wind_speed).
    Column is typically windSpeedMean."""
    time_col = 'startDateTime'
    wind_col = None
    for c in ['windSpeedMean']:
        if c in df.columns:
            wind_col = c
            break
    if wind_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'windspeed' in cl and 'mean' in cl:
                wind_col = c
                break
    if wind_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'wind' in cl and 'mean' in cl \
               and 'expuncert' not in cl and 'finalqf' not in cl:
                wind_col = c
                break
    if wind_col is None:
        raise ValueError(f"No wind column. Cols: {list(df.columns)}")

    print(f"  Wind speed column: {wind_col}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    out = df[[time_col, wind_col]].copy()
    out.columns = ['datetime', 'wind_speed']
    out = out.dropna(subset=['wind_speed'])
    out['wind_speed'] = out['wind_speed'].clip(lower=0)
    out = out.set_index('datetime').resample('30min').mean().dropna()
    return out.reset_index()


# ==============================================================================
#  MERGE ALL FOUR STREAMS
# ==============================================================================

def merge_all(df_water, df_par, df_air, df_wind):
    """Inner-join all four 30-min DataFrames on datetime. Strict: drop NAs."""
    print("\n--- Merging 4 data streams ---")
    print(f"  Water temp: {len(df_water)} rows")
    print(f"  PAR:        {len(df_par)} rows")
    print(f"  Air temp:   {len(df_air)} rows")
    print(f"  Wind speed: {len(df_wind)} rows")

    # Start with water temp as base
    df = df_water.copy()

    # Merge PAR
    df = pd.merge_asof(
        df.sort_values('datetime'),
        df_par.sort_values('datetime'),
        on='datetime',
        tolerance=pd.Timedelta('35min'),
        direction='nearest'
    )

    # Merge air temp
    df = pd.merge_asof(
        df.sort_values('datetime'),
        df_air.sort_values('datetime'),
        on='datetime',
        tolerance=pd.Timedelta('35min'),
        direction='nearest'
    )

    # Merge wind speed
    df = pd.merge_asof(
        df.sort_values('datetime'),
        df_wind.sort_values('datetime'),
        on='datetime',
        tolerance=pd.Timedelta('35min'),
        direction='nearest'
    )

    # Strict: drop ALL rows with any NaN in the four core columns
    n_before = len(df)
    df = df.dropna(subset=['water_temp', 'air_temp', 'wind_speed', 'par'])
    df = df.reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"  After strict NA drop: {len(df)} rows ({n_dropped} dropped)")

    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} clean rows after merge (need >= 100). "
            "Check data availability."
        )

    # Compute timestamp (seconds from start) and time_delta
    t0 = df['datetime'].iloc[0]
    df['timestamp'] = (df['datetime'] - t0).dt.total_seconds()
    dt_seconds = df['datetime'].diff().dt.total_seconds()
    dt_seconds.iloc[0] = DT
    df['time_delta'] = dt_seconds

    # Final output columns
    df_out = df[['timestamp', 'time_delta', 'water_temp',
                 'air_temp', 'wind_speed', 'par']].copy()

    print(f"\n  Final dataset: {len(df_out)} samples")
    print(f"  Time span: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    for col in ['water_temp', 'air_temp', 'wind_speed', 'par']:
        print(f"    {col}: mean={df_out[col].mean():.3f}, "
              f"std={df_out[col].std():.3f}, "
              f"range=[{df_out[col].min():.3f}, {df_out[col].max():.3f}]")

    return df_out, df['datetime'].iloc[0], df['datetime'].iloc[-1]


# ==============================================================================
#  CREATE SPLITS
# ==============================================================================

def create_splits(df):
    """Chronological train/val/test splits."""
    print("\n--- Creating Temporal Splits ---")
    N = len(df)
    n_train = int(N * TRAIN_FRAC)
    n_val = int(N * VAL_FRAC)

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    for name, dfs in [('train', df_train), ('val', df_val), ('test', df_test)]:
        path = SPLITS_DIR / f"{name}.csv"
        dfs.to_csv(path, index=False)
        print(f"  {name}: {len(dfs)} samples -> {path}")

    return df_train, df_val, df_test


# ==============================================================================
#  MAIN
# ==============================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("NEON Lake Temperature Benchmark: Data Acquisition v2")
    print("  Real data only -- no synthetic fallbacks")
    print("=" * 70)
    print(f"Months: {YEAR_MONTHS}")
    print(f"dt: {DT}s ({DT/3600:.1f}h)")
    print(f"Products:")
    for key, prod in PRODUCTS.items():
        print(f"  {key}: {prod}")

    # --- Step 1: Find a site with all 4 products ---
    site = select_site()

    # --- Step 2: Download all 4 products ---
    df_water_raw = download_product(
        PRODUCTS['water_temp'], site, YEAR_MONTHS,
        'water_temp', ['TSD_30_min', 'TSD_30min', 'TSD'],
        multi_sensor=True)

    df_par_raw = download_product(
        PRODUCTS['par'], site, YEAR_MONTHS,
        'par', ['PARPAR_30min', '30min', 'PAR'])

    df_air_raw = download_product(
        PRODUCTS['buoy_air'], site, YEAR_MONTHS,
        'buoy_air', ['RHbuoy_30min', '30min', 'RHbuoy'])

    df_wind_raw = download_product(
        PRODUCTS['wind'], site, YEAR_MONTHS,
        'wind', ['2DWSD_30min', '30min', 'wind'])

    # Validate all four downloaded
    for name, df in [('water_temp', df_water_raw), ('par', df_par_raw),
                     ('buoy_air', df_air_raw), ('wind', df_wind_raw)]:
        if df is None or len(df) == 0:
            raise RuntimeError(f"Failed to download {name} for site {site}")

    # --- Step 3: Parse each product ---
    print("\n--- Parsing products ---")
    df_water = parse_water_temp(df_water_raw)
    df_par = parse_par(df_par_raw)
    df_air = parse_buoy_air_temp(df_air_raw)
    df_wind = parse_wind(df_wind_raw)

    # --- Step 4: Merge ---
    df_clean, dt_start, dt_end = merge_all(df_water, df_par, df_air, df_wind)

    # --- Step 5: Save ---
    processed_path = PROCESSED_DIR / f"neon_{site}_processed.csv"
    df_clean.to_csv(processed_path, index=False)
    print(f"\n  Saved: {processed_path}")
    print(f"  SHA-256: {sha256_file(processed_path)}")

    # --- Step 6: Create splits ---
    df_train, df_val, df_test = create_splits(df_clean)

    # --- Step 7: Manifest ---
    manifest = {
        'site': site,
        'source': 'NEON API (4 real products)',
        'products': PRODUCTS,
        'year_months': YEAR_MONTHS,
        'dt_seconds': DT,
        'time_range': [str(dt_start), str(dt_end)],
        'n_total': len(df_clean),
        'n_train': len(df_train),
        'n_val': len(df_val),
        'n_test': len(df_test),
        'processed_csv': str(processed_path),
        'processed_sha256': sha256_file(processed_path),
        'split_hashes': {
            'train': sha256_file(SPLITS_DIR / "train.csv"),
            'val': sha256_file(SPLITS_DIR / "val.csv"),
            'test': sha256_file(SPLITS_DIR / "test.csv"),
        },
        'columns': list(df_clean.columns),
        'stats': {
            col: {
                'mean': float(df_clean[col].mean()),
                'std': float(df_clean[col].std()),
                'min': float(df_clean[col].min()),
                'max': float(df_clean[col].max()),
            } for col in ['water_temp', 'air_temp', 'wind_speed', 'par']
        },
    }

    manifest_path = BENCH_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
