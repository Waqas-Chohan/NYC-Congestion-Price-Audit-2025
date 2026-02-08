"""
Ghost Trip Filter Module
Detects and filters suspicious/invalid taxi trips based on business rules
"""

import sys
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import Tuple
import logging

# Add parent directory to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MAX_SPEED_MPH, MIN_TRIP_DURATION_MINUTES, TELEPORTER_FARE_THRESHOLD,
    STATIONARY_DISTANCE, STATIONARY_FARE_MIN, AUDIT_LOGS_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_trip_speed(df: dd.DataFrame, 
                         pickup_col: str = 'pickup_datetime',
                         dropoff_col: str = 'dropoff_datetime',
                         distance_col: str = 'trip_distance') -> dd.Series:
    """
    Calculate trip speed in miles per hour.
    
    Args:
        df: Dask DataFrame with trip data
        pickup_col: Name of pickup datetime column
        dropoff_col: Name of dropoff datetime column
        distance_col: Name of distance column
        
    Returns:
        Dask Series with speed in mph
    """
    # Calculate trip duration in hours
    trip_duration_hours = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 3600
    
    # Avoid division by zero
    trip_duration_hours = trip_duration_hours.replace(0, np.nan)
    
    # Calculate speed
    speed_mph = df[distance_col] / trip_duration_hours
    
    return speed_mph


def detect_impossible_physics(df: dd.DataFrame) -> dd.Series:
    """
    Detect trips with impossible physics (speed > 65 mph).
    
    Args:
        df: Dask DataFrame with trip data
        
    Returns:
        Boolean Series indicating impossible physics trips
    """
    speed = calculate_trip_speed(df)
    return speed > MAX_SPEED_MPH


def detect_teleporter(df: dd.DataFrame,
                     pickup_col: str = 'pickup_datetime',
                     dropoff_col: str = 'dropoff_datetime',
                     fare_col: str = 'fare_amount') -> dd.Series:
    """
    Detect "teleporter" trips: trip time < 1 minute but fare > $20.
    
    Args:
        df: Dask DataFrame with trip data
        pickup_col: Name of pickup datetime column
        dropoff_col: Name of dropoff datetime column
        fare_col: Name of fare amount column
        
    Returns:
        Boolean Series indicating teleporter trips
    """
    # Calculate trip duration in minutes
    trip_duration_minutes = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    
    # Detect teleporters: < 1 minute AND fare > $20
    is_teleporter = (trip_duration_minutes < MIN_TRIP_DURATION_MINUTES) & \
                    (df[fare_col] > TELEPORTER_FARE_THRESHOLD)
    
    return is_teleporter


def detect_stationary_ride(df: dd.DataFrame,
                           distance_col: str = 'trip_distance',
                           fare_col: str = 'fare_amount') -> dd.Series:
    """
    Detect stationary rides: trip distance = 0 but fare > 0.
    
    Args:
        df: Dask DataFrame with trip data
        distance_col: Name of distance column
        fare_col: Name of fare amount column
        
    Returns:
        Boolean Series indicating stationary rides
    """
    is_stationary = (df[distance_col] == STATIONARY_DISTANCE) & \
                   (df[fare_col] > STATIONARY_FARE_MIN)
    
    return is_stationary


def apply_ghost_trip_filters(df: dd.DataFrame,
                             taxi_type: str = 'unknown') -> Tuple[dd.DataFrame, dd.DataFrame]:
    """
    Apply all ghost trip filters and separate clean from dirty data.
    
    Args:
        df: Dask DataFrame with trip data
        taxi_type: Type of taxi ('yellow', 'green', etc.)
        
    Returns:
        Tuple of (clean_data, ghost_trips)
    """
    logger.info(f"Applying ghost trip filters to {taxi_type} taxi data...")
    
    # Create filter columns
    df = df.assign(
        is_impossible_physics=detect_impossible_physics(df),
        is_teleporter=detect_teleporter(df),
        is_stationary=detect_stationary_ride(df)
    )
    
    # Combined ghost trip flag
    df = df.assign(
        is_ghost_trip=df['is_impossible_physics'] | 
                     df['is_teleporter'] | 
                     df['is_stationary']
    )
    
    # Separate clean and ghost trips
    clean_data = df[~df['is_ghost_trip']]
    ghost_trips = df[df['is_ghost_trip']]
    
    # Log statistics
    total_trips = len(df)
    ghost_count = len(ghost_trips)
    
    # Note: These computations will be lazy until computed
    logger.info(f"  Total trips: {total_trips}")
    logger.info(f"  Ghost trips detected: {ghost_count}")
    logger.info(f"  Clean trips: {total_trips - ghost_count}")
    
    return clean_data, ghost_trips


def save_ghost_trip_audit_log(ghost_trips: dd.DataFrame,
                               taxi_type: str,
                               month: str,
                               compute: bool = True) -> str:
    """
    Save ghost trip audit log to parquet file.
    
    Args:
        ghost_trips: Dask DataFrame with ghost trip data
        taxi_type: Type of taxi
        month: Month identifier (e.g., '2025-01')
        compute: If True, compute and save immediately
        
    Returns:
        Path to saved audit log
    """
    # Create audit log filename
    audit_log_path = AUDIT_LOGS_DIR / f"ghost_trips_{taxi_type}_{month}.parquet"
    
    logger.info(f"Saving ghost trip audit log to: {audit_log_path}")
    
    if compute:
        # Compute and save
        ghost_trips.to_parquet(
            str(audit_log_path),
            engine='pyarrow',
            compression='snappy'
        )
    else:
        # Return delayed save operation
        return ghost_trips.to_parquet(
            str(audit_log_path),
            engine='pyarrow',
            compression='snappy',
            compute=False
        )
    
    logger.info(f"✓ Ghost trip audit log saved")
    return str(audit_log_path)


def generate_ghost_trip_summary(ghost_trips: dd.DataFrame,
                                taxi_type: str) -> pd.DataFrame:
    """
    Generate summary statistics for ghost trips.
    
    Args:
        ghost_trips: Dask DataFrame with ghost trip data
        taxi_type: Type of taxi
        
    Returns:
        Pandas DataFrame with summary statistics
    """
    logger.info(f"Generating ghost trip summary for {taxi_type}...")
    
    # Count by violation type
    summary = pd.DataFrame({
        'Taxi Type': [taxi_type],
        'Total Ghost Trips': [len(ghost_trips)],
        'Impossible Physics': [ghost_trips['is_impossible_physics'].sum().compute()],
        'Teleporter': [ghost_trips['is_teleporter'].sum().compute()],
        'Stationary Ride': [ghost_trips['is_stationary'].sum().compute()]
    })
    
    # Top 5 vendors by ghost trip count
    if 'vendor_id' in ghost_trips.columns:
        vendor_counts = ghost_trips.groupby('vendor_id').size().compute()
        vendor_counts = vendor_counts.sort_values(ascending=False).head(5)
        
        logger.info(f"\nTop 5 Suspicious Vendors ({taxi_type}):")
        for vendor_id, count in vendor_counts.items():
            logger.info(f"  Vendor {vendor_id}: {count} ghost trips")
    
    return summary


def filter_and_audit_data(input_path: str,
                          output_clean_path: str,
                          taxi_type: str,
                          month: str) -> dict:
    """
    Complete pipeline to filter ghost trips and save audit logs.
    
    Args:
        input_path: Path to input parquet file
        output_clean_path: Path to save cleaned data
        taxi_type: Type of taxi
        month: Month identifier
        
    Returns:
        Dictionary with filtering statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Ghost Trip Filtering: {taxi_type} - {month}")
    logger.info(f"{'='*60}\n")
    
    # Read data
    logger.info(f"Reading data from: {input_path}")
    df = dd.read_parquet(input_path)
    
    # Apply filters
    clean_data, ghost_trips = apply_ghost_trip_filters(df, taxi_type)
    
    # Compute counts
    total_count = len(df)
    clean_count = len(clean_data)
    ghost_count = len(ghost_trips)
    
    # Save clean data
    logger.info(f"Saving clean data to: {output_clean_path}")
    clean_data.to_parquet(
        output_clean_path,
        engine='pyarrow',
        compression='snappy'
    )
    
    # Save ghost trip audit log
    save_ghost_trip_audit_log(ghost_trips, taxi_type, month)
    
    # Generate summary
    summary = generate_ghost_trip_summary(ghost_trips, taxi_type)
    
    # Return statistics
    stats = {
        'total_trips': total_count,
        'clean_trips': clean_count,
        'ghost_trips': ghost_count,
        'ghost_percentage': (ghost_count / total_count * 100) if total_count > 0 else 0,
        'summary': summary
    }
    
    logger.info(f"\n✓ Filtering complete:")
    logger.info(f"  Clean trips: {clean_count:,} ({100-stats['ghost_percentage']:.2f}%)")
    logger.info(f"  Ghost trips: {ghost_count:,} ({stats['ghost_percentage']:.2f}%)")
    
    return stats


if __name__ == "__main__":
    """
    Test the ghost trip filter on sample data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Ghost Trip Filter")
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--output', required=True, help='Output clean data path')
    parser.add_argument('--taxi-type', default='yellow', help='Taxi type')
    parser.add_argument('--month', default='2025-01', help='Month identifier')
    
    args = parser.parse_args()
    
    stats = filter_and_audit_data(
        input_path=args.input,
        output_clean_path=args.output,
        taxi_type=args.taxi_type,
        month=args.month
    )
    
    print("\nFiltering Statistics:")
    print(stats['summary'])
