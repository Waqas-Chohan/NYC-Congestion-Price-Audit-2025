"""
Weather API Module
Handles fetching and processing weather data from OpenMeteo API
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging
import json

# Add parent directory to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    WEATHER_API_URL, CENTRAL_PARK_LATITUDE, CENTRAL_PARK_LONGITUDE,
    WEATHER_START_DATE, WEATHER_END_DATE, WEATHER_VARIABLES,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDataFetcher:
    """
    Class to fetch and cache weather data from OpenMeteo API.
    """
    
    def __init__(self, cache_dir: Path = None):
        """
        Initialize weather data fetcher.
        
        Args:
            cache_dir: Directory to cache weather data
        """
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_file = self.cache_dir / "weather_data_2025.csv"
        self.weather_df = None
    
    def fetch_weather_data(self,
                          start_date: str = WEATHER_START_DATE,
                          end_date: str = WEATHER_END_DATE,
                          latitude: float = CENTRAL_PARK_LATITUDE,
                          longitude: float = CENTRAL_PARK_LONGITUDE) -> pd.DataFrame:
        """
        Fetch daily weather data from OpenMeteo API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            DataFrame with daily weather data
        """
        logger.info("Fetching weather data from OpenMeteo API...")
        logger.info(f"  Location: Central Park, NYC ({latitude}, {longitude})")
        logger.info(f"  Date range: {start_date} to {end_date}")
        
        # Construct API request
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(['precipitation_sum', 'temperature_2m_mean', 
                              'temperature_2m_max', 'temperature_2m_min']),
            'timezone': 'America/New_York'
        }
        
        try:
            # Make API request
            response = requests.get(WEATHER_API_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Extract daily data
            daily = data['daily']
            
            # Create DataFrame
            weather_df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'precipitation_mm': daily['precipitation_sum'],
                'temperature_mean_c': daily['temperature_2m_mean'],
                'temperature_max_c': daily['temperature_2m_max'],
                'temperature_min_c': daily['temperature_2m_min']
            })
            
            logger.info(f"✓ Successfully fetched {len(weather_df)} days of weather data")
            
            self.weather_df = weather_df
            return weather_df
            
        except requests.RequestException as e:
            logger.error(f"✗ Error fetching weather data: {e}")
            raise
        except KeyError as e:
            logger.error(f"✗ Unexpected API response format: {e}")
            raise
    
    def save_to_cache(self, df: pd.DataFrame = None) -> str:
        """
        Save weather data to cache file.
        
        Args:
            df: DataFrame to save (uses self.weather_df if None)
            
        Returns:
            Path to cached file
        """
        if df is None:
            df = self.weather_df
        
        if df is None:
            raise ValueError("No weather data to save")
        
        logger.info(f"Saving weather data to cache: {self.cache_file}")
        df.to_csv(self.cache_file, index=False)
        logger.info(f"✓ Weather data cached")
        
        return str(self.cache_file)
    
    def load_from_cache(self) -> pd.DataFrame:
        """
        Load weather data from cache file.
        
        Returns:
            DataFrame with cached weather data
        """
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_file}")
        
        logger.info(f"Loading weather data from cache: {self.cache_file}")
        self.weather_df = pd.read_csv(self.cache_file, parse_dates=['date'])
        logger.info(f"✓ Loaded {len(self.weather_df)} days from cache")
        
        return self.weather_df
    
    def get_weather_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get weather data, from cache if available, otherwise fetch from API.
        
        Args:
            use_cache: If True, try to load from cache first
            
        Returns:
            DataFrame with weather data
        """
        if use_cache and self.cache_file.exists():
            try:
                return self.load_from_cache()
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                logger.info("Fetching fresh data from API...")
        
        # Fetch from API
        weather_df = self.fetch_weather_data()
        
        # Save to cache
        self.save_to_cache(weather_df)
        
        return weather_df
    
    def get_rainy_days(self, threshold_mm: float = 1.0) -> pd.DataFrame:
        """
        Get days with precipitation above threshold.
        
        Args:
            threshold_mm: Minimum precipitation in mm to be considered rainy
            
        Returns:
            DataFrame with rainy days
        """
        if self.weather_df is None:
            self.get_weather_data()
        
        rainy_days = self.weather_df[
            self.weather_df['precipitation_mm'] >= threshold_mm
        ].copy()
        
        logger.info(f"Found {len(rainy_days)} rainy days (>= {threshold_mm}mm)")
        
        return rainy_days
    
    def get_wettest_month(self) -> Tuple[int, float]:
        """
        Identify the wettest month based on total precipitation.
        
        Returns:
            Tuple of (month_number, total_precipitation_mm)
        """
        if self.weather_df is None:
            self.get_weather_data()
        
        # Extract month
        monthly_precip = self.weather_df.copy()
        monthly_precip['month'] = monthly_precip['date'].dt.month
        
        # Sum precipitation by month
        monthly_totals = monthly_precip.groupby('month')['precipitation_mm'].sum()
        
        # Find wettest month
        wettest_month = monthly_totals.idxmax()
        wettest_precip = monthly_totals.max()
        
        logger.info(f"Wettest month: {wettest_month} ({wettest_precip:.1f}mm total)")
        
        return int(wettest_month), float(wettest_precip)
    
    def get_weather_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for weather data.
        
        Returns:
            DataFrame with weather summary
        """
        if self.weather_df is None:
            self.get_weather_data()
        
        summary = pd.DataFrame({
            'Metric': [
                'Total Days',
                'Rainy Days (>1mm)',
                'Total Precipitation (mm)',
                'Average Daily Precip (mm)',
                'Max Daily Precip (mm)',
                'Average Temperature (°C)',
                'Coldest Day (°C)',
                'Hottest Day (°C)'
            ],
            'Value': [
                len(self.weather_df),
                len(self.weather_df[self.weather_df['precipitation_mm'] >= 1.0]),
                self.weather_df['precipitation_mm'].sum(),
                self.weather_df['precipitation_mm'].mean(),
                self.weather_df['precipitation_mm'].max(),
                self.weather_df['temperature_mean_c'].mean(),
                self.weather_df['temperature_min_c'].min(),
                self.weather_df['temperature_max_c'].max()
            ]
        })
        
        return summary


def calculate_rain_elasticity(trip_counts: pd.DataFrame,
                              weather_data: pd.DataFrame,
                              date_col: str = 'date',
                              count_col: str = 'trip_count',
                              precip_col: str = 'precipitation_mm') -> Dict:
    """
    Calculate rain elasticity of demand.
    
    Args:
        trip_counts: DataFrame with daily trip counts
        weather_data: DataFrame with daily weather data
        date_col: Name of date column
        count_col: Name of trip count column
        precip_col: Name of precipitation column
        
    Returns:
        Dictionary with elasticity metrics
    """
    logger.info("Calculating rain elasticity of demand...")
    
    # Merge trip counts with weather data
    combined = pd.merge(
        trip_counts,
        weather_data[[date_col, precip_col]],
        on=date_col,
        how='inner'
    )
    
    # Calculate correlation
    correlation = combined[count_col].corr(combined[precip_col])
    
    # Determine elasticity interpretation
    if correlation < -0.3:
        interpretation = "Elastic (rain significantly reduces demand)"
    elif correlation < -0.1:
        interpretation = "Moderately elastic"
    elif correlation < 0.1:
        interpretation = "Inelastic (rain has little effect)"
    else:
        interpretation = "Positive correlation (unusual - may indicate other factors)"
    
    # Calculate additional metrics
    rainy_days = combined[combined[precip_col] >= 1.0]
    dry_days = combined[combined[precip_col] < 1.0]
    
    avg_trips_rainy = rainy_days[count_col].mean() if len(rainy_days) > 0 else 0
    avg_trips_dry = dry_days[count_col].mean() if len(dry_days) > 0 else 0
    
    pct_change = ((avg_trips_rainy - avg_trips_dry) / avg_trips_dry * 100) if avg_trips_dry > 0 else 0
    
    results = {
        'correlation': correlation,
        'interpretation': interpretation,
        'avg_trips_rainy_days': avg_trips_rainy,
        'avg_trips_dry_days': avg_trips_dry,
        'pct_change_on_rainy_days': pct_change,
        'rainy_days_count': len(rainy_days),
        'dry_days_count': len(dry_days)
    }
    
    logger.info(f"Rain Elasticity Score: {correlation:.3f} ({interpretation})")
    logger.info(f"Average trips on rainy days: {avg_trips_rainy:,.0f}")
    logger.info(f"Average trips on dry days: {avg_trips_dry:,.0f}")
    logger.info(f"Percentage change: {pct_change:+.1f}%")
    
    return results


if __name__ == "__main__":
    """
    Test the weather module.
    """
    # Initialize fetcher
    fetcher = WeatherDataFetcher()
    
    # Fetch weather data
    try:
        weather_df = fetcher.get_weather_data(use_cache=True)
        
        print("\n" + "="*60)
        print("WEATHER DATA SUMMARY (NYC 2025)")
        print("="*60)
        summary = fetcher.get_weather_summary()
        print(summary.to_string(index=False))
        
        print("\n" + "="*60)
        print("WETTEST MONTH")
        print("="*60)
        month_num, total_precip = fetcher.get_wettest_month()
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        print(f"Month: {month_names[month_num]}")
        print(f"Total Precipitation: {total_precip:.1f}mm")
        
        print("\n" + "="*60)
        print("SAMPLE DATA (First 5 days)")
        print("="*60)
        print(weather_df.head().to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print("\nMake sure you have an internet connection to fetch weather data.")
