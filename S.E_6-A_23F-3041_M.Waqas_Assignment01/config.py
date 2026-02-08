"""
Configuration file for NYC Congestion Pricing Audit
Contains all paths, constants, and API configurations
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Base directory (project root)
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUDIT_LOGS_DIR = DATA_DIR / "audit_logs"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
VISUALS_DIR = OUTPUTS_DIR / "visuals"

# Ensure all directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, AUDIT_LOGS_DIR, VISUALS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SOURCE URLS
# ============================================================================
# TLC Trip Record Data
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
TLC_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TLC_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# ============================================================================
# DATA SCHEMA
# ============================================================================
# Unified schema columns to keep
UNIFIED_SCHEMA = [
    'tpep_pickup_datetime',   # Yellow taxi
    'tpep_dropoff_datetime',  # Yellow taxi
    'lpep_pickup_datetime',   # Green taxi
    'lpep_dropoff_datetime',  # Green taxi
    'PULocationID',
    'DOLocationID',
    'trip_distance',
    'fare_amount',
    'total_amount',
    'tip_amount',
    'congestion_surcharge',
    'VendorID'
]

# Standardized output columns
OUTPUT_COLUMNS = [
    'pickup_datetime',
    'dropoff_datetime',
    'pickup_location_id',
    'dropoff_location_id',
    'trip_distance',
    'fare_amount',
    'total_amount',
    'tip_amount',
    'congestion_surcharge',
    'vendor_id',
    'taxi_type'  # 'yellow' or 'green'
]

# ============================================================================
# CONGESTION ZONE CONSTANTS
# ============================================================================
# Manhattan congestion relief zone (south of 60th Street)
# These are the LocationIDs from the TLC taxi zone lookup
# Source: NYC TLC Taxi Zone Maps - Manhattan below 60th St
CONGESTION_ZONE_IDS = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137,
    140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163,
    164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232,
    233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263
]

# Congestion pricing start date
CONGESTION_START_DATE = "2025-01-05"

# ============================================================================
# GHOST TRIP FILTER THRESHOLDS
# ============================================================================
# Maximum realistic speed in mph
MAX_SPEED_MPH = 65

# Minimum trip duration for high fare (in minutes)
MIN_TRIP_DURATION_MINUTES = 1
TELEPORTER_FARE_THRESHOLD = 20  # dollars

# Stationary ride threshold
STATIONARY_DISTANCE = 0
STATIONARY_FARE_MIN = 0  # Any fare > 0 for zero distance is suspicious

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Date ranges for analysis
Q1_2024_START = "2024-01-01"
Q1_2024_END = "2024-03-31"
Q1_2025_START = "2025-01-01"
Q1_2025_END = "2025-03-31"

YEAR_2024 = 2024
YEAR_2025 = 2025

# Months for data collection (January to November for 2025)
MONTHS_TO_DOWNLOAD = list(range(1, 12))  # 1-11 (Jan-Nov)

# ============================================================================
# WEATHER API CONFIGURATION
# ============================================================================
# OpenMeteo API (free, no API key required)
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Central Park coordinates (NYC)
CENTRAL_PARK_LATITUDE = 40.7829
CENTRAL_PARK_LONGITUDE = -73.9654

# Weather data parameters
WEATHER_START_DATE = "2025-01-01"
WEATHER_END_DATE = "2025-12-31"
WEATHER_VARIABLES = ["precipitation_sum", "temperature_2m_mean"]

# Rain elasticity threshold
RAIN_ELASTICITY_THRESHOLD = -0.3  # Elastic demand if correlation < -0.3

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# Heatmap parameters
HOURS_OF_DAY = list(range(24))  # 0-23
DAYS_OF_WEEK = list(range(7))   # 0-6 (Monday-Sunday)

# Color schemes for visualizations
COLORMAP_HEATMAP = "RdYlGn"
COLORMAP_CHOROPLETH = "RdYlBu_r"

# ============================================================================
# DASK CONFIGURATION
# ============================================================================
# Dask chunking parameters for memory efficiency
DASK_CHUNKSIZE = "64MB"
DASK_NPARTITIONS = 10

# ============================================================================
# REPORTING
# ============================================================================
# Executive report file
AUDIT_REPORT_PATH = OUTPUTS_DIR / "audit_report.pdf"

# Dashboard file
DASHBOARD_PATH = OUTPUTS_DIR / "dashboard.py"

# ============================================================================
# IMPUTATION WEIGHTS (for missing December 2025 data)
# ============================================================================
DEC_2023_WEIGHT = 0.3
DEC_2024_WEIGHT = 0.7

# ============================================================================
# TAXI TYPES
# ============================================================================
TAXI_TYPES = {
    'yellow': 'yellow_tripdata',
    'green': 'green_tripdata'
}

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
