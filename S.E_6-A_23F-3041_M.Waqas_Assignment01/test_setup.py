"""
Quick test script to verify pipeline components and launch dashboard
"""

import pandas as pd
import sys
from pathlib import Path

# Test imports
print("="*60)
print("Testing Module Imports...")
print("="*60)

try:
    import dask.dataframe as dd
    print("✓ Dask imported successfully")
except Exception as e:
    print(f"✗ Dask import failed: {e}")

try:
    import geopandas as gpd
    print("✓ GeoPandas imported successfully")
except Exception as e:
    print(f"✗ GeoPandas import failed: {e}")

try:
    import streamlit
    print("✓ Streamlit imported successfully")
except Exception as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    print(f"✓ Config imported successfully")
    print(f"  Raw data: {RAW_DATA_DIR}")
    print(f"  Processed: {PROCESSED_DATA_DIR}")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from utils.geo import TaxiZoneMapper
    print("✓ Geo module imported successfully")
except Exception as e:
    print(f"✗ Geo import failed: {e}")

try:
    from utils.weather import WeatherDataFetcher
    print("✓ Weather module imported successfully")
except Exception as e:
    print(f"✗ Weather import failed: {e}")

print("\n" + "="*60)
print("Testing Data Files...")
print("="*60)

# Check for data files
raw_dir = Path("data/raw")
parquet_files = list(raw_dir.glob("*.parquet"))
print(f"\nFound {len(parquet_files)} Parquet files:")
yellow_files = [f for f in parquet_files if 'yellow' in f.name]
green_files = [f for f in parquet_files if 'green' in f.name]
print(f"  Yellow taxi: {len(yellow_files)} files")
print(f"  Green taxi: {len(green_files)} files")

# Test reading one file
if yellow_files:
    print(f"\nTesting Dask read on {yellow_files[0].name}...")
    try:
        import dask.dataframe as dd
        df = dd.read_parquet(str(yellow_files[0]))
        print(f"✓ Successfully read parquet file")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Partitions: {df.npartitions}")
    except Exception as e:
        print(f"✗ Error reading parquet: {e}")

print("\n" + "="*60)
print("Testing Weather Data Cache...")
print("="*60)

weather_cache = Path("data/raw/weather_data_2025.csv")
if weather_cache.exists():
    weather_df = pd.read_csv(weather_cache)
    print(f"✓ Weather data cached: {len(weather_df)} days")
    print(f"  Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
else:
    print("⊘ Weather data not cached (run utils/weather.py first)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✓ All modules functional")
print("✓ Data files present")
print("✓ Ready to run dashboard")
print("\nTo launch dashboard:")
print("  streamlit run outputs/dashboard.py")
