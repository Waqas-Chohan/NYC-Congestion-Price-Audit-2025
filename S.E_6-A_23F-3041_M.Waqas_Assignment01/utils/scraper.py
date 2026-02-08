"""
Web Scraping Module for NYC TLC Trip Data
Automates downloading of taxi trip data, zone lookups, and shapefiles
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Tuple
import time
from tqdm import tqdm
import logging

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TLC_BASE_URL, TLC_ZONE_LOOKUP_URL, TLC_ZONE_SHAPEFILE_URL,
    RAW_DATA_DIR, TAXI_TYPES, MONTHS_TO_DOWNLOAD,
    DEC_2023_WEIGHT, DEC_2024_WEIGHT, YEAR_2024, YEAR_2025
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_url_exists(url: str) -> bool:
    """
    Check if a URL exists by sending a HEAD request.
    
    Args:
        url: The URL to check
        
    Returns:
        True if URL exists (status code 200), False otherwise
    """
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.warning(f"Error checking URL {url}: {e}")
        return False


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Send GET request
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Successfully downloaded: {destination.name}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"✗ Error downloading {url}: {e}")
        return False


def download_month(taxi_type: str, year: int, month: int, 
                   force_download: bool = False) -> Tuple[bool, str]:
    """
    Download taxi trip data for a specific month.
    
    Args:
        taxi_type: Either 'yellow' or 'green'
        year: Year (e.g., 2025)
        month: Month (1-12)
        force_download: If True, re-download even if file exists
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Construct filename and URL
    filename = f"{TAXI_TYPES[taxi_type]}_{year}-{month:02d}.parquet"
    url = f"{TLC_BASE_URL}{filename}"
    destination = RAW_DATA_DIR / filename
    
    # Check if file already exists
    if destination.exists() and not force_download:
        logger.info(f"⊘ File already exists: {filename}")
        return True, f"Already exists: {filename}"
    
    # Check if URL exists
    logger.info(f"Checking availability: {filename}")
    if not check_url_exists(url):
        return False, f"File not available: {filename}"
    
    # Download the file
    logger.info(f"Downloading: {filename}")
    success = download_file(url, destination)
    
    if success:
        return True, f"Downloaded: {filename}"
    else:
        return False, f"Failed to download: {filename}"


def download_all_months(taxi_type: str, year: int, 
                        months: List[int] = MONTHS_TO_DOWNLOAD,
                        force_download: bool = False) -> dict:
    """
    Download trip data for all specified months.
    
    Args:
        taxi_type: Either 'yellow' or 'green'
        year: Year to download
        months: List of months (1-12)
        force_download: If True, re-download existing files
        
    Returns:
        Dictionary with download results
    """
    results = {
        'successful': [],
        'failed': [],
        'existing': []
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {taxi_type.upper()} taxi data for {year}")
    logger.info(f"{'='*60}\n")
    
    for month in months:
        success, message = download_month(taxi_type, year, month, force_download)
        
        if success:
            if "Already exists" in message:
                results['existing'].append((year, month))
            else:
                results['successful'].append((year, month))
        else:
            results['failed'].append((year, month))
        
        # Be nice to the server - small delay between downloads
        time.sleep(1)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary for {taxi_type.upper()} {year}:")
    logger.info(f"  ✓ Successfully downloaded: {len(results['successful'])}")
    logger.info(f"  ⊘ Already existing: {len(results['existing'])}")
    logger.info(f"  ✗ Failed: {len(results['failed'])}")
    logger.info(f"{'='*60}\n")
    
    return results


def download_taxi_zone_lookup(force_download: bool = False) -> bool:
    """
    Download the taxi zone lookup CSV file.
    
    Args:
        force_download: If True, re-download even if exists
        
    Returns:
        True if successful, False otherwise
    """
    destination = RAW_DATA_DIR / "taxi_zone_lookup.csv"
    
    if destination.exists() and not force_download:
        logger.info(f"⊘ Taxi zone lookup already exists")
        return True
    
    logger.info("Downloading taxi zone lookup CSV...")
    return download_file(TLC_ZONE_LOOKUP_URL, destination)


def download_taxi_zone_shapefile(force_download: bool = False) -> bool:
    """
    Download the taxi zone shapefile (as ZIP).
    
    Args:
        force_download: If True, re-download even if exists
        
    Returns:
        True if successful, False otherwise
    """
    destination = RAW_DATA_DIR / "taxi_zones.zip"
    
    if destination.exists() and not force_download:
        logger.info(f"⊘ Taxi zone shapefile already exists")
        return True
    
    logger.info("Downloading taxi zone shapefile...")
    success = download_file(TLC_ZONE_SHAPEFILE_URL, destination)
    
    if success:
        # Extract the ZIP file
        import zipfile
        try:
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR / "taxi_zones")
            logger.info("✓ Shapefile extracted successfully")
        except zipfile.BadZipFile:
            logger.error("✗ Error extracting shapefile")
            return False
    
    return success


def impute_december_2025(taxi_type: str) -> bool:
    """
    Impute December 2025 data using weighted average of Dec 2023 and Dec 2024.
    Formula: Dec_2025 = 0.3 * Dec_2023 + 0.7 * Dec_2024
    
    NOTE: This implementation creates a symbolic reference file.
    Actual imputation logic would be implemented in the data processing pipeline
    where Dask/PySpark can handle the large-scale computation.
    
    Args:
        taxi_type: Either 'yellow' or 'green'
        
    Returns:
        True if imputation setup is successful
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Setting up December 2025 imputation for {taxi_type}")
    logger.info(f"{'='*60}\n")
    
    # Check if December 2025 data exists
    dec_2025_file = RAW_DATA_DIR / f"{TAXI_TYPES[taxi_type]}_2025-12.parquet"
    
    if dec_2025_file.exists():
        logger.info("✓ December 2025 data already exists. No imputation needed.")
        return True
    
    # Check for required files (Dec 2023 and Dec 2024)
    dec_2023_file = RAW_DATA_DIR / f"{TAXI_TYPES[taxi_type]}_2023-12.parquet"
    dec_2024_file = RAW_DATA_DIR / f"{TAXI_TYPES[taxi_type]}_2024-12.parquet"
    
    missing_files = []
    if not dec_2023_file.exists():
        missing_files.append("December 2023")
    if not dec_2024_file.exists():
        missing_files.append("December 2024")
    
    if missing_files:
        logger.warning(f"⚠ Missing required files for imputation: {', '.join(missing_files)}")
        logger.info("Downloading required historical data...")
        
        # Download Dec 2023 if needed
        if "December 2023" in missing_files:
            download_month(taxi_type, 2023, 12)
        
        # Download Dec 2024 if needed
        if "December 2024" in missing_files:
            download_month(taxi_type, 2024, 12)
    
    # Create imputation marker file
    # Actual imputation will be done in the main pipeline using Dask
    imputation_marker = RAW_DATA_DIR / f"{TAXI_TYPES[taxi_type]}_2025-12_IMPUTED.txt"
    with open(imputation_marker, 'w') as f:
        f.write(f"December 2025 imputation metadata\n")
        f.write(f"Formula: Dec_2025 = {DEC_2023_WEIGHT} * Dec_2023 + {DEC_2024_WEIGHT} * Dec_2024\n")
        f.write(f"Source files:\n")
        f.write(f"  - {dec_2023_file.name}\n")
        f.write(f"  - {dec_2024_file.name}\n")
        f.write(f"\nImputation will be performed in the main pipeline.\n")
    
    logger.info(f"✓ Imputation setup complete. Marker file created.")
    logger.info(f"  Actual imputation will be done in the data processing pipeline.")
    
    return True


def run_full_scraping_pipeline(force_download: bool = False):
    """
    Run the complete scraping pipeline:
    1. Download zone lookup and shapefiles
    2. Download yellow taxi data (Jan-Nov 2025)
    3. Download green taxi data (Jan-Nov 2025)
    4. Setup December 2025 imputation
    
    Args:
        force_download: If True, re-download all files
    """
    logger.info("\n" + "="*60)
    logger.info("NYC TAXI DATA SCRAPING PIPELINE")
    logger.info("="*60 + "\n")
    
    # Step 1: Download metadata files
    logger.info("STEP 1: Downloading metadata files...")
    download_taxi_zone_lookup(force_download)
    download_taxi_zone_shapefile(force_download)
    
    # NOTE: Yellow and Green taxi data for Jan-Nov 2025 are already downloaded
    # The following code is COMMENTED OUT but represents the working scraping logic
    
    # # Step 2: Download Yellow taxi data
    # logger.info("\nSTEP 2: Downloading Yellow taxi data...")
    # yellow_results = download_all_months('yellow', YEAR_2025, 
    #                                      MONTHS_TO_DOWNLOAD, force_download)
    
    # # Step 3: Download Green taxi data
    # logger.info("\nSTEP 3: Downloading Green taxi data...")
    # green_results = download_all_months('green', YEAR_2025,
    #                                     MONTHS_TO_DOWNLOAD, force_download)
    
    logger.info("\n⊘ SKIPPING Yellow/Green taxi downloads (already have data)")
    logger.info("  If you need to download them, uncomment lines in scraper.py")
    
    # Step 4: Setup December imputation
    logger.info("\nSTEP 4: Setting up December 2025 imputation...")
    impute_december_2025('yellow')
    impute_december_2025('green')
    
    logger.info("\n" + "="*60)
    logger.info("SCRAPING PIPELINE COMPLETE")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    """
    Run scraper as standalone script.
    Usage: python scraper.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="NYC TLC Data Scraper")
    parser.add_argument('--force', action='store_true',
                       help='Force re-download of existing files')
    parser.add_argument('--taxi-type', choices=['yellow', 'green', 'both'],
                       default='both', help='Which taxi type to download')
    parser.add_argument('--year', type=int, default=YEAR_2025,
                       help='Year to download')
    parser.add_argument('--metadata-only', action='store_true',
                       help='Only download zone lookup and shapefiles')
    
    args = parser.parse_args()
    
    if args.metadata_only:
        download_taxi_zone_lookup(args.force)
        download_taxi_zone_shapefile(args.force)
    else:
        run_full_scraping_pipeline(args.force)
