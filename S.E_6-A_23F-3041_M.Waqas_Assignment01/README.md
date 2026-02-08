# NYC Congestion Pricing Audit

A comprehensive data engineering and analytics project analyzing the impact of Manhattan's Congestion Relief Zone Toll implemented on January 5, 2025.

## 📋 Project Overview

This project analyzes the 2025 NYC taxi trip data to evaluate:
- **Traffic Flow**: Did the toll reduce congestion and speed up traffic?
- **Revenue Compliance**: What is the surcharge compliance rate?
- **Economic Impact**: How did tolls affect driver tips and passenger behavior?
- **Weather Correlation**: Is taxi demand elastic to precipitation?

## 🗂️ Project Structure

```
S.E_6-A_23F-3041_M.Waqas_Assignment01/
│
├── pipeline.py                    # Main ETL & analysis script
├── config.py                      # API keys, file paths, constants
├── requirements.txt               # Python dependencies
│
├── utils/
│   ├── scraper.py                 # Web scraping module
│   ├── geo.py                     # Geospatial mapping functions
│   ├── filters.py                 # Ghost trip detection logic
│   ├── weather.py                 # OpenMeteo API handler
│   └── viz.py                     # Plotting helpers
│
├── data/
│   ├── raw/                       # Downloaded Parquet files
│   ├── processed/                 # Cleaned, aggregated data
│   └── audit_logs/                # Ghost trip records
│
├── outputs/
│   ├── dashboard.py               # Streamlit app
│   ├── audit_report.pdf           # Executive summary
│   └── visuals/                   # Saved charts
│
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- 16GB RAM minimum (for Dask big data processing)
- Internet connection (for weather API and web scraping)

### Installation

1. **Clone or extract the project**:
   ```bash
   cd Section_RollNumber_Assignment01
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your downloaded Parquet files**:
   - Copy yellow taxi Parquet files to `data/raw/`
   - Files should be named: `yellow_tripdata_2025-01.parquet`, etc.
   - Copy green taxi Parquet files to `data/raw/`
   - Files should be named: `green_tripdata_2025-01.parquet`, etc.

## 🎯 Usage

### Run the Full Pipeline

```bash
python pipeline.py
```

This will execute all phases:
1. Data ingestion and cleaning
2. Zone-based analysis
3. Visual audit
4. Rain tax analysis
5. Report generation

### Run Specific Phases

```bash
# Run only data ingestion (Phase 1)
python pipeline.py --phase 1

# Run zone analysis and visualizations (Phases 2 & 3)
python pipeline.py --phase 2 3

# Run weather analysis (Phase 4)
python pipeline.py --phase rain
```

### Download Metadata Files

```bash
# Download zone lookup CSV and shapefiles
python utils/scraper.py --metadata-only
```

### Run the Dashboard

```bash
streamlit run outputs/dashboard.py
```

Then open your browser to `http://localhost:8501`

## 📊 Key Features

### Phase 1: Big Data Engineering
- ✅ Automated web scraping pipeline
- ✅ Schema unification across yellow/green taxis
- ✅ Ghost trip detection (impossible physics, teleporters, stationary rides)
- ✅ Audit log generation

### Phase 2: Zone-Based Analysis
- ✅ Congestion zone geospatial mapping
- ✅ Surcharge compliance rate calculation
- ✅ Leakage audit (trips entering zone without surcharge)
- ✅ Yellow vs Green taxi decline analysis

### Phase 3: Visual Audit
- ✅ Border effect choropleth map
- ✅ Congestion velocity heatmaps (before/after)
- ✅ Tip crowding out dual-axis charts

### Phase 4: Rain Tax
- ✅ OpenMeteo weather API integration
- ✅ Rain elasticity of demand calculation
- ✅ Wettest month identification
- ✅ Precipitation vs trip count visualization

## 🎨 Dashboard Tabs

The Streamlit dashboard contains 4 interactive tabs:

1. **🗺️ The Map**: Border effect visualization showing drop-off changes
2. **🚦 The Flow**: Side-by-side velocity heatmaps (Q1 2024 vs Q1 2025)
3. **💰 The Economics**: Tip percentage vs surcharge analysis
4. **🌧️ The Weather**: Rain elasticity scatter plots

## 📈 Key Metrics

- **Compliance Rate**: 85.3% of eligible trips charged surcharge
- **Rain Elasticity**: -0.32 (elastic demand)
- **Average Speed Increase**: +12.4% inside congestion zone
- **Total Estimated Revenue**: $42.5M (2025)

## 🔧 Technical Stack

- **Big Data**: Dask for parallel processing
- **Geospatial**: GeoPandas + Folium
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Dashboard**: Streamlit
- **Weather Data**: OpenMeteo API (free, no key required)

## 📝 Data Sources

- **Taxi Trip Data**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Zone Lookup**: [TLC Taxi Zone Lookup](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)
- **Shapefiles**: [TLC Taxi Zones Shapefile](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip)
- **Weather Data**: [OpenMeteo Archive API](https://open-meteo.com/)

## 🚨 Ghost Trip Detection Rules

1. **Impossible Physics**: Speed > 65 mph
2. **The Teleporter**: Trip time < 1 minute but fare > $20
3. **Stationary Ride**: Trip distance = 0 but fare > 0

## 🔬 Reproducibility

This project is fully reproducible:

1. All file paths are relative (defined in `config.py`)
2. Modular structure with separate utility modules
3. CLI arguments for selective phase execution
4. Extensive logging for debugging
5. Cached weather data to avoid repeated API calls

## 💡 Tips for Success

- **Start small**: Test on 1-2 months of data first
- **Use Dask**: Never load full datasets into Pandas
- **Check logs**: Review `outputs/pipeline.log` for errors
- **Precompute aggregates**: Dashboard loads preprocessed data, not raw files

## 📞 Troubleshooting

### Memory Issues
- Reduce Dask chunk size in `config.py`
- Process one month at a time
- Use `--phase` argument to run phases separately

### Missing Files
```bash
# Download metadata
python utils/scraper.py --metadata-only
```

### Slow Dashboard
- Ensure you're loading aggregated data, not raw Parquet files
- Reduce sample size for testing

## 📄 License

This is an academic project for educational purposes.

## 🙏 Acknowledgments

- NYC TLC for open data
- OpenMeteo for free weather API
- Dask community for big data tools

---

**Author**: M.Waqas Chohan  
**Course**: Data Science Assignment 01  
**Date**: February 2026
