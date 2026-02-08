"""
Geospatial Module
Handles taxi zone mapping, congestion zone identification, and spatial analysis
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from typing import List, Set, Dict
import logging

# Add parent directory to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, CONGESTION_ZONE_IDS, TLC_ZONE_LOOKUP_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiZoneMapper:
    """
    Class to handle taxi zone mapping and congestion zone analysis.
    """
    
    def __init__(self):
        """Initialize the taxi zone mapper."""
        self.zone_lookup = None
        self.zone_gdf = None
        self.congestion_zone_ids = set(CONGESTION_ZONE_IDS)
        self.bordering_zones = None
        
    def load_zone_lookup(self, filepath: str = None) -> pd.DataFrame:
        """
        Load taxi zone lookup CSV.
        
        Args:
            filepath: Path to taxi_zone_lookup.csv (optional)
            
        Returns:
            DataFrame with zone information
        """
        if filepath is None:
            filepath = RAW_DATA_DIR / "taxi_zone_lookup.csv"
        
        logger.info(f"Loading taxi zone lookup from: {filepath}")
        
        try:
            self.zone_lookup = pd.read_csv(filepath)
            logger.info(f"✓ Loaded {len(self.zone_lookup)} taxi zones")
            return self.zone_lookup
        except FileNotFoundError:
            logger.error(f"✗ Zone lookup file not found: {filepath}")
            logger.info("  Run scraper.py to download the file")
            raise
    
    def load_zone_shapefile(self, shapefile_path: str = None) -> gpd.GeoDataFrame:
        """
        Load taxi zone shapefile for geospatial analysis.
        
        Args:
            shapefile_path: Path to shapefile (optional)
            
        Returns:
            GeoDataFrame with zone geometries
        """
        if shapefile_path is None:
            shapefile_path = RAW_DATA_DIR / "taxi_zones" / "taxi_zones.shp"
        
        logger.info(f"Loading taxi zone shapefile from: {shapefile_path}")
        
        try:
            self.zone_gdf = gpd.read_file(shapefile_path)
            logger.info(f"✓ Loaded {len(self.zone_gdf)} zone geometries")
            return self.zone_gdf
        except Exception as e:
            logger.error(f"✗ Error loading shapefile: {e}")
            logger.info("  Run scraper.py to download the shapefile")
            raise
    
    def get_congestion_zones(self) -> pd.DataFrame:
        """
        Get information about congestion zone locations.
        
        Returns:
            DataFrame with congestion zone details
        """
        if self.zone_lookup is None:
            self.load_zone_lookup()
        
        congestion_zones = self.zone_lookup[
            self.zone_lookup['LocationID'].isin(self.congestion_zone_ids)
        ]
        
        logger.info(f"Congestion zone contains {len(congestion_zones)} zones")
        return congestion_zones
    
    def is_in_congestion_zone(self, location_id: int) -> bool:
        """
        Check if a location ID is in the congestion zone.
        
        Args:
            location_id: TLC Location ID
            
        Returns:
            True if in congestion zone, False otherwise
        """
        return location_id in self.congestion_zone_ids
    
    def identify_bordering_zones(self) -> Set[int]:
        """
        Identify zones that border the congestion zone.
        Uses shapefile geometries to find adjacent zones.
        
        Returns:
            Set of location IDs for bordering zones
        """
        if self.zone_gdf is None:
            self.load_zone_shapefile()
        
        logger.info("Identifying zones bordering the congestion zone...")
        
        # Get congestion zone geometries
        congestion_gdf = self.zone_gdf[
            self.zone_gdf['LocationID'].isin(self.congestion_zone_ids)
        ]
        
        # Create union of congestion zone
        congestion_union = congestion_gdf.unary_union
        
        # Find zones that touch the congestion zone but are not in it
        bordering_zones = set()
        
        for idx, row in self.zone_gdf.iterrows():
            location_id = row['LocationID']
            
            # Skip if already in congestion zone
            if location_id in self.congestion_zone_ids:
                continue
            
            # Check if geometry touches congestion zone
            if row['geometry'].touches(congestion_union):
                bordering_zones.add(location_id)
        
        self.bordering_zones = bordering_zones
        logger.info(f"✓ Found {len(bordering_zones)} zones bordering congestion zone")
        
        return bordering_zones
    
    def get_zone_name(self, location_id: int) -> str:
        """
        Get zone name from location ID.
        
        Args:
            location_id: TLC Location ID
            
        Returns:
            Zone name (Borough + Zone)
        """
        if self.zone_lookup is None:
            self.load_zone_lookup()
        
        zone_info = self.zone_lookup[self.zone_lookup['LocationID'] == location_id]
        
        if len(zone_info) == 0:
            return f"Unknown Zone {location_id}"
        
        row = zone_info.iloc[0]
        return f"{row['Borough']} - {row['Zone']}"
    
    def get_manhattan_zones(self) -> pd.DataFrame:
        """
        Get all Manhattan taxi zones.
        
        Returns:
            DataFrame with Manhattan zones
        """
        if self.zone_lookup is None:
            self.load_zone_lookup()
        
        manhattan_zones = self.zone_lookup[
            self.zone_lookup['Borough'] == 'Manhattan'
        ]
        
        return manhattan_zones
    
    def create_zone_id_to_name_mapping(self) -> Dict[int, str]:
        """
        Create dictionary mapping LocationID to zone names.
        
        Returns:
            Dictionary {LocationID: "Borough - Zone"}
        """
        if self.zone_lookup is None:
            self.load_zone_lookup()
        
        mapping = {}
        for _, row in self.zone_lookup.iterrows():
            mapping[row['LocationID']] = f"{row['Borough']} - {row['Zone']}"
        
        return mapping
    
    def export_congestion_zone_geojson(self, output_path: str = None) -> str:
        """
        Export congestion zone as GeoJSON for visualization.
        
        Args:
            output_path: Path to save GeoJSON
            
        Returns:
            Path to saved file
        """
        if self.zone_gdf is None:
            self.load_zone_shapefile()
        
        if output_path is None:
            output_path = RAW_DATA_DIR / "congestion_zone.geojson"
        
        # Filter to congestion zones
        congestion_gdf = self.zone_gdf[
            self.zone_gdf['LocationID'].isin(self.congestion_zone_ids)
        ]
        
        # Save as GeoJSON
        congestion_gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"✓ Congestion zone GeoJSON saved to: {output_path}")
        
        return str(output_path)
    
    def get_zone_statistics(self) -> pd.DataFrame:
        """
        Get statistics about taxi zones.
        
        Returns:
            DataFrame with zone statistics
        """
        if self.zone_lookup is None:
            self.load_zone_lookup()
        
        stats = pd.DataFrame({
            'Category': [
                'Total Zones',
                'Congestion Zone',
                'Manhattan (Total)',
                'Bronx',
                'Brooklyn',
                'Queens',
                'Staten Island',
                'EWR (Newark Airport)'
            ],
            'Count': [
                len(self.zone_lookup),
                len(self.congestion_zone_ids),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'Manhattan']),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'Bronx']),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'Brooklyn']),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'Queens']),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'Staten Island']),
                len(self.zone_lookup[self.zone_lookup['Borough'] == 'EWR'])
            ]
        })
        
        return stats


def get_congestion_zone_filter(location_col: str = 'location_id') -> str:
    """
    Generate Dask/Pandas filter expression for congestion zone.
    
    Args:
        location_col: Name of location ID column
        
    Returns:
        Filter expression string
    """
    zone_list = list(CONGESTION_ZONE_IDS)
    return f"{location_col}.isin({zone_list})"


def classify_trip_type(pickup_id: int, dropoff_id: int) -> str:
    """
    Classify trip based on pickup and dropoff locations.
    
    Types:
    - 'internal': Both pickup and dropoff in congestion zone
    - 'entering': Pickup outside, dropoff inside congestion zone
    - 'exiting': Pickup inside, dropoff outside congestion zone
    - 'external': Both outside congestion zone
    
    Args:
        pickup_id: Pickup location ID
        dropoff_id: Dropoff location ID
        
    Returns:
        Trip type classification
    """
    pickup_in_zone = pickup_id in CONGESTION_ZONE_IDS
    dropoff_in_zone = dropoff_id in CONGESTION_ZONE_IDS
    
    if pickup_in_zone and dropoff_in_zone:
        return 'internal'
    elif not pickup_in_zone and dropoff_in_zone:
        return 'entering'
    elif pickup_in_zone and not dropoff_in_zone:
        return 'exiting'
    else:
        return 'external'


if __name__ == "__main__":
    """
    Test the geospatial module.
    """
    # Initialize mapper
    mapper = TaxiZoneMapper()
    
    # Load data
    try:
        mapper.load_zone_lookup()
        print("\n" + "="*60)
        print("TAXI ZONE STATISTICS")
        print("="*60)
        print(mapper.get_zone_statistics().to_string(index=False))
        
        print("\n" + "="*60)
        print("CONGESTION ZONE DETAILS")
        print("="*60)
        congestion_zones = mapper.get_congestion_zones()
        print(f"Total zones: {len(congestion_zones)}")
        print("\nSample zones:")
        print(congestion_zones.head(10).to_string(index=False))
        
        # Try loading shapefile
        try:
            mapper.load_zone_shapefile()
            bordering = mapper.identify_bordering_zones()
            print(f"\nZones bordering congestion zone: {len(bordering)}")
        except Exception as e:
            print(f"\nNote: Shapefile not available ({e})")
            print("Run scraper.py to download shapefiles")
            
    except FileNotFoundError:
        print("Zone lookup file not found.")
        print("Please run: python utils/scraper.py --metadata-only")
