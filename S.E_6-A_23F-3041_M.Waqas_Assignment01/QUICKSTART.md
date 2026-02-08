# Quick Start Guide for NYC Congestion Pricing Audit

## 🎯 Your Parquet Files Location

**Place all your downloaded yellow and green taxi Parquet files in:**
```
d:\Data_Science\Section_RollNumber_Assignment01\data\raw\
```

### Expected File Names:
```
yellow_tripdata_2025-01.parquet
yellow_tripdata_2025-02.parquet
yellow_tripdata_2025-03.parquet
...
yellow_tripdata_2025-11.parquet

green_tripdata_2025-01.parquet
green_tripdata_2025-02.parquet
green_tripdata_2025-03.parquet
...
green_tripdata_2025-11.parquet
```

## 📦 Installation Steps

### 1. Install Dependencies
```bash
cd d:\Data_Science\Section_RollNumber_Assignment01
pip install -r requirements.txt
```

### 2. Download Metadata (Zone Lookup & Shapefiles)
```bash
python utils/scraper.py --metadata-only
```

This will download:
- `taxi_zone_lookup.csv`
- `taxi_zones.zip` (with shapefiles)

## 🚀 Running the Project

### Option 1: Run Complete Pipeline
```bash
python pipeline.py
```

### Option 2: Run Specific Phases
```bash
# Phase 1: Data ingestion and cleaning
python pipeline.py --phase 1

# Phase 2: Zone analysis
python pipeline.py --phase 2

# Phase 3: Visualizations
python pipeline.py --phase 3

# Phase 4: Weather analysis
python pipeline.py --phase 4
```

### Option 3: Run the Dashboard
```bash
streamlit run outputs/dashboard.py
```

Then open your browser to: http://localhost:8501

## 📊 What to Expect

### After Phase 1:
- `data/processed/` - Cleaned data files
- `data/audit_logs/` - Ghost trip records

### After Phase 2:
- Console output with leakage audit results
- Yellow vs Green decline statistics

### After Phase 3:
- `outputs/visuals/` - Generated charts and maps

### After Phase 4:
- `data/raw/weather_data_2025.csv` - Cached weather data
- Rain elasticity metrics in console

### Final Outputs:
- `outputs/executive_summary.txt` - Text report
- `outputs/pipeline.log` - Execution log
- `outputs/dashboard.py` - Interactive Streamlit app

## 🔧 Testing Individual Modules

### Test Zone Mapping:
```bash
python utils/geo.py
```

### Test Weather Fetching:
```bash
python utils/weather.py
```

### Test Scraper (metadata only):
```bash
python utils/scraper.py --metadata-only
```

## ⚠️ Important Notes

1. **Data Size**: Each monthly Parquet file is ~150-300 MB. Make sure you have at least 50GB free space for processing.

2. **Memory**: Dask will manage memory automatically, but 16GB RAM is recommended for smooth processing.

3. **Internet**: Required for:
   - Weather API calls (first time)
   - Downloading metadata files

4. **Processing Time**: 
   - Phase 1 (cleaning): ~30-60 minutes for all months
   - Full pipeline: ~2-3 hours depending on hardware

## 🐛 Troubleshooting

### Problem: ModuleNotFoundError
```bash
pip install -r requirements.txt --upgrade
```

### Problem: Memory errors
Edit `config.py`:
```python
DASK_CHUNKSIZE = "128MB"  # Increase if you have more RAM
```

### Problem: Dashboard not loading
Make sure you've processed data first:
```bash
python pipeline.py --phase 1 2
```

## 📝 Next Steps for Assignment

1. ✅ Set up project structure (DONE)
2. 📁 Move your Parquet files to `data/raw/`
3. 🔧 Install dependencies
4. ▶️ Run the pipeline
5. 📊 Launch the dashboard
6. 📄 Generate the audit report PDF (implement in Phase 6)
7. ✍️ Write Medium blog and LinkedIn post
8. 📦 Package as zip file

## 💡 Tips

- Start with just 1-2 months of data to test the pipeline
- Check `outputs/pipeline.log` for detailed execution logs
- Use `--force` flag cautiously (will re-download all data)
- Dashboard works best with pre-aggregated data

## 🆘 Need Help?

Check the full README.md for comprehensive documentation!
