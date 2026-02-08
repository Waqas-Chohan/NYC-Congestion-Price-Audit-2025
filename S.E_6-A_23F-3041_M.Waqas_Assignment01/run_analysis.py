"""
NYC Congestion Pricing Audit - Complete Analysis Script
Implements Phase 2, 3, and 4 with real data processing
Run this to generate all analysis outputs and visualizations
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALS_DIR = OUTPUT_DIR / "visuals"

# Create output directories
VISUALS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Congestion zone IDs (Manhattan south of 60th St - 69 zones)
CONGESTION_ZONE_IDS = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137,
    140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163,
    164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232,
    233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263
]

print("=" * 70)
print("NYC CONGESTION PRICING AUDIT - COMPLETE ANALYSIS")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_yellow_taxi():
    """Load all yellow taxi data."""
    files = list(RAW_DATA_DIR.glob("yellow_tripdata_2025-*.parquet"))
    print(f"Loading {len(files)} yellow taxi files...")
    ddf = dd.read_parquet([str(f) for f in sorted(files)])
    return ddf

def load_green_taxi():
    """Load all green taxi data."""
    files = list(RAW_DATA_DIR.glob("green_tripdata_2025-*.parquet"))
    print(f"Loading {len(files)} green taxi files...")
    ddf = dd.read_parquet([str(f) for f in sorted(files)])
    return ddf

def load_zone_lookup():
    """Load taxi zone lookup."""
    zone_file = RAW_DATA_DIR / "taxi_zone_lookup.csv"
    return pd.read_csv(zone_file)

def load_weather_data():
    """Load weather data."""
    weather_file = RAW_DATA_DIR / "weather_data_2025.csv"
    if weather_file.exists():
        return pd.read_csv(weather_file, parse_dates=['date'])
    return None

# ============================================================================
# PHASE 2: ZONE-BASED ANALYSIS
# ============================================================================

def phase2_leakage_audit(yellow_ddf):
    """
    Phase 2a: Leakage Audit Analysis
    - Calculate surcharge compliance rate
    - Identify top 3 pickup locations with missing surcharges
    """
    print("\n" + "=" * 70)
    print("PHASE 2a: LEAKAGE AUDIT ANALYSIS")
    print("=" * 70)
    
    # Check for congestion_surcharge column
    if 'congestion_surcharge' not in yellow_ddf.columns:
        print("Warning: congestion_surcharge column not found")
        return None
    
    # Convert to pandas for easier processing (compute subset)
    # Get trips in congestion zone with key columns
    cols = ['PULocationID', 'DOLocationID', 'congestion_surcharge']
    sample_df = yellow_ddf[cols].compute()
    
    # Filter to congestion zone
    congestion_mask = (
        sample_df['PULocationID'].isin(CONGESTION_ZONE_IDS) |
        sample_df['DOLocationID'].isin(CONGESTION_ZONE_IDS)
    )
    congestion_trips = sample_df[congestion_mask]
    
    total_congestion_trips = len(congestion_trips)
    
    # Trips with surcharge applied (> 0)
    trips_with_surcharge = len(congestion_trips[congestion_trips['congestion_surcharge'] > 0])
    
    # Trips with missing surcharge
    trips_missing_surcharge = total_congestion_trips - trips_with_surcharge
    
    compliance_rate = (trips_with_surcharge / total_congestion_trips * 100) if total_congestion_trips > 0 else 0
    
    print(f"\nSurcharge Compliance Analysis:")
    print(f"  Total trips in/out of congestion zone: {total_congestion_trips:,}")
    print(f"  Trips with surcharge applied: {trips_with_surcharge:,}")
    print(f"  Trips missing surcharge: {trips_missing_surcharge:,}")
    print(f"  Compliance Rate: {compliance_rate:.1f}%")
    print(f"  Leakage Rate: {100 - compliance_rate:.1f}%")
    
    # Top 3 pickup locations with missing surcharges
    print("\nTop 3 Pickup Locations with Missing Surcharges:")
    missing_surcharge_trips = congestion_trips[
        (congestion_trips['congestion_surcharge'] == 0) | 
        (congestion_trips['congestion_surcharge'].isna())
    ]
    
    missing_by_location = missing_surcharge_trips.groupby('PULocationID').size()
    top3_missing = missing_by_location.nlargest(3)
    
    zone_lookup = load_zone_lookup()
    for i, (loc_id, count) in enumerate(top3_missing.items(), 1):
        zone_name = zone_lookup[zone_lookup['LocationID'] == loc_id]['Zone'].values
        zone_name = zone_name[0] if len(zone_name) > 0 else "Unknown"
        print(f"  {i}. Zone {loc_id} ({zone_name}): {count:,} trips missing surcharge")
    
    # Save results
    results = {
        'total_congestion_trips': int(total_congestion_trips),
        'trips_with_surcharge': int(trips_with_surcharge),
        'trips_missing_surcharge': int(trips_missing_surcharge),
        'compliance_rate': float(compliance_rate),
        'leakage_rate': float(100 - compliance_rate),
        'top3_missing_locations': [
            {'location_id': int(loc_id), 'missing_count': int(count)}
            for loc_id, count in top3_missing.items()
        ]
    }
    
    with open(OUTPUT_DIR / "leakage_audit_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: outputs/leakage_audit_results.json")
    return results


def phase2_yellow_vs_green(yellow_ddf, green_ddf):
    """
    Phase 2b: Yellow vs Green Decline Analysis
    - Compare trip volumes by month
    """
    print("\n" + "=" * 70)
    print("PHASE 2b: YELLOW VS GREEN DECLINE ANALYSIS")
    print("=" * 70)
    
    # Get monthly counts for yellow - compute first then process
    yellow_df = yellow_ddf[['tpep_pickup_datetime']].compute()
    yellow_df['month'] = pd.to_datetime(yellow_df['tpep_pickup_datetime']).dt.month
    yellow_monthly = yellow_df.groupby('month').size()
    
    # Get monthly counts for green
    green_df = green_ddf[['lpep_pickup_datetime']].compute()
    green_df['month'] = pd.to_datetime(green_df['lpep_pickup_datetime']).dt.month
    green_monthly = green_df.groupby('month').size()
    
    # Create comparison dataframe
    months = sorted(set(yellow_monthly.index) | set(green_monthly.index))
    comparison = pd.DataFrame({
        'month': months,
        'yellow_trips': [yellow_monthly.get(m, 0) for m in months],
        'green_trips': [green_monthly.get(m, 0) for m in months]
    })
    
    comparison['total_trips'] = comparison['yellow_trips'] + comparison['green_trips']
    comparison['yellow_pct'] = comparison['yellow_trips'] / comparison['total_trips'] * 100
    comparison['green_pct'] = comparison['green_trips'] / comparison['total_trips'] * 100
    
    print("\nMonthly Trip Volume Comparison:")
    print("-" * 60)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, row in comparison.iterrows():
        m = int(row['month']) - 1
        if m < len(month_names):
            print(f"  {month_names[m]}: Yellow={row['yellow_trips']:>10,}  Green={row['green_trips']:>8,}  Total={row['total_trips']:>10,}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Total Yellow Trips (2025): {comparison['yellow_trips'].sum():,}")
    print(f"  Total Green Trips (2025): {comparison['green_trips'].sum():,}")
    print(f"  Yellow Market Share: {comparison['yellow_trips'].sum() / comparison['total_trips'].sum() * 100:.1f}%")
    print(f"  Green Market Share: {comparison['green_trips'].sum() / comparison['total_trips'].sum() * 100:.1f}%")
    
    # Q1 Analysis
    q1_yellow = comparison[comparison['month'] <= 3]['yellow_trips'].sum()
    q1_green = comparison[comparison['month'] <= 3]['green_trips'].sum()
    print(f"\nQ1 2025 (Jan-Mar) Analysis:")
    print(f"  Yellow: {q1_yellow:,} trips")
    print(f"  Green: {q1_green:,} trips")
    
    # Create visualization
    comparison['month_name'] = comparison['month'].apply(lambda x: month_names[int(x)-1] if x <= 12 else 'Dec')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison['month_name'],
        y=comparison['yellow_trips'],
        name='Yellow Taxi',
        marker_color='gold'
    ))
    
    fig.add_trace(go.Bar(
        x=comparison['month_name'],
        y=comparison['green_trips'],
        name='Green Taxi',
        marker_color='forestgreen'
    ))
    
    fig.update_layout(
        title='Yellow vs Green Taxi Monthly Trip Volume (2025)',
        xaxis_title='Month',
        yaxis_title='Number of Trips',
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.write_html(str(VISUALS_DIR / "yellow_vs_green_comparison.html"))
    fig.write_image(str(VISUALS_DIR / "yellow_vs_green_comparison.png"), scale=2)
    
    print(f"\nVisualization saved to: outputs/visuals/yellow_vs_green_comparison.html")
    
    # Save data
    comparison.to_csv(OUTPUT_DIR / "yellow_green_comparison.csv", index=False)
    
    return comparison


# ============================================================================
# PHASE 3: VISUAL AUDIT
# ============================================================================

def phase3_border_effect(yellow_ddf):
    """
    Phase 3a: Border Effect Choropleth
    """
    print("\n" + "=" * 70)
    print("PHASE 3a: BORDER EFFECT ANALYSIS")
    print("=" * 70)
    
    # Count drop-offs by zone - compute first
    dropoff_df = yellow_ddf[['DOLocationID']].compute()
    dropoff_counts = dropoff_df.groupby('DOLocationID').size()
    
    zone_lookup = load_zone_lookup()
    
    # Create dataframe
    border_data = pd.DataFrame({
        'zone_id': dropoff_counts.index,
        'dropoff_count': dropoff_counts.values
    })
    
    border_data = border_data.merge(zone_lookup, left_on='zone_id', right_on='LocationID', how='left')
    
    # Label zones
    border_data['zone_type'] = 'Other'
    border_data.loc[border_data['zone_id'].isin(CONGESTION_ZONE_IDS), 'zone_type'] = 'Congestion Zone'
    
    # Summary by zone type
    print("\nDrop-off Analysis by Zone Type:")
    zone_summary = border_data.groupby('zone_type').agg({
        'dropoff_count': ['sum', 'mean', 'count']
    }).round(0)
    zone_summary.columns = ['Total Drop-offs', 'Avg per Zone', 'Num Zones']
    print(zone_summary)
    
    # Top 10 drop-off zones
    print("\nTop 10 Drop-off Zones:")
    top10 = border_data.nlargest(10, 'dropoff_count')[['zone_id', 'Zone', 'Borough', 'dropoff_count', 'zone_type']]
    for i, row in top10.iterrows():
        zone_name = str(row['Zone'])[:30] if pd.notna(row['Zone']) else "Unknown"
        borough = str(row['Borough'])[:15] if pd.notna(row['Borough']) else "Unknown"
        print(f"  {zone_name:30} ({borough:15}): {row['dropoff_count']:>10,} - {row['zone_type']}")
    
    # Create bar chart
    top20 = border_data.nlargest(20, 'dropoff_count').sort_values('dropoff_count', ascending=True)
    
    fig = px.bar(
        top20,
        x='dropoff_count',
        y='Zone',
        color='zone_type',
        orientation='h',
        title='Top 20 Drop-off Zones by Volume (2025)',
        labels={'dropoff_count': 'Number of Drop-offs', 'Zone': 'Taxi Zone'},
        color_discrete_map={
            'Congestion Zone': '#FF6B6B',
            'Other': '#95A5A6'
        },
        height=600
    )
    
    fig.write_html(str(VISUALS_DIR / "border_effect_dropoffs.html"))
    fig.write_image(str(VISUALS_DIR / "border_effect_dropoffs.png"), scale=2)
    
    print(f"\nVisualization saved to: outputs/visuals/border_effect_dropoffs.html")
    
    # Save data
    border_data.to_csv(OUTPUT_DIR / "border_effect_data.csv", index=False)
    
    return border_data


def phase3_velocity_heatmap(yellow_ddf):
    """
    Phase 3b: Congestion Velocity Heatmap
    """
    print("\n" + "=" * 70)
    print("PHASE 3b: VELOCITY HEATMAP ANALYSIS")
    print("=" * 70)
    
    # Get relevant columns and compute
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID']
    df = yellow_ddf[cols].compute()
    
    # Filter to congestion zone
    congestion_mask = (
        df['PULocationID'].isin(CONGESTION_ZONE_IDS) |
        df['DOLocationID'].isin(CONGESTION_ZONE_IDS)
    )
    congestion_trips = df[congestion_mask].copy()
    
    # Calculate trip duration
    congestion_trips['trip_duration_hours'] = (
        (pd.to_datetime(congestion_trips['tpep_dropoff_datetime']) - 
         pd.to_datetime(congestion_trips['tpep_pickup_datetime'])).dt.total_seconds() / 3600
    )
    
    # Filter valid trips
    valid_trips = congestion_trips[
        (congestion_trips['trip_duration_hours'] > 0.01) &
        (congestion_trips['trip_duration_hours'] < 3) &
        (congestion_trips['trip_distance'] > 0)
    ].copy()
    
    # Calculate speed
    valid_trips['speed_mph'] = valid_trips['trip_distance'] / valid_trips['trip_duration_hours']
    valid_trips['hour'] = pd.to_datetime(valid_trips['tpep_pickup_datetime']).dt.hour
    valid_trips['day_of_week'] = pd.to_datetime(valid_trips['tpep_pickup_datetime']).dt.dayofweek
    
    # Filter reasonable speeds
    valid_trips = valid_trips[(valid_trips['speed_mph'] > 0) & (valid_trips['speed_mph'] < 65)]
    
    # Aggregate
    velocity_agg = valid_trips.groupby(['hour', 'day_of_week'])['speed_mph'].mean().reset_index()
    
    # Pivot for heatmap
    heatmap_data = velocity_agg.pivot(index='day_of_week', columns='hour', values='speed_mph')
    
    print("\nAverage Speed Statistics in Congestion Zone:")
    print(f"  Overall Average: {velocity_agg['speed_mph'].mean():.1f} mph")
    peak_hours = velocity_agg[(velocity_agg['hour'] >= 8) & (velocity_agg['hour'] <= 9)]
    if len(peak_hours) > 0:
        print(f"  Peak Hour (8-9am) Avg: {peak_hours['speed_mph'].mean():.1f} mph")
    off_peak = velocity_agg[(velocity_agg['hour'] >= 22) | (velocity_agg['hour'] <= 6)]
    if len(off_peak) > 0:
        print(f"  Off-Peak (10pm-6am) Avg: {off_peak['speed_mph'].mean():.1f} mph")
    
    # Create heatmap with matplotlib
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        annot=True,
        fmt='.1f',
        cbar_kws={'label': 'Average Speed (mph)'},
        yticklabels=day_labels
    )
    plt.title('Average Trip Speed in Congestion Zone by Hour and Day (2025)', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(str(VISUALS_DIR / "velocity_heatmap.png"), dpi=150)
    plt.close()
    
    # Create Plotly version
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Speed (mph)"),
        title="Average Trip Speed in Congestion Zone (2025)",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        height=500
    )
    fig.update_yaxes(ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], tickvals=list(range(7)))
    fig.write_html(str(VISUALS_DIR / "velocity_heatmap.html"))
    
    print(f"\nVisualization saved to: outputs/visuals/velocity_heatmap.html")
    
    # Save data
    velocity_agg.to_csv(OUTPUT_DIR / "velocity_data.csv", index=False)
    
    return velocity_agg


def phase3_tip_crowding_out(yellow_ddf):
    """
    Phase 3c: Tip Crowding Out Analysis
    """
    print("\n" + "=" * 70)
    print("PHASE 3c: TIP CROWDING OUT ANALYSIS")
    print("=" * 70)
    
    # Get relevant columns
    cols = ['tpep_pickup_datetime', 'fare_amount', 'tip_amount']
    if 'congestion_surcharge' in yellow_ddf.columns:
        cols.append('congestion_surcharge')
    
    df = yellow_ddf[cols].compute()
    
    # Filter valid fares
    df = df[df['fare_amount'] > 0].copy()
    
    # Calculate tip percentage
    df['tip_pct'] = (df['tip_amount'] / df['fare_amount'] * 100).clip(0, 100)
    df['month'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.month
    
    # Handle congestion_surcharge
    if 'congestion_surcharge' not in df.columns:
        df['congestion_surcharge'] = 0
    
    # Aggregate
    monthly = df.groupby('month').agg({
        'tip_pct': 'mean',
        'congestion_surcharge': 'mean',
        'tip_amount': 'mean',
        'fare_amount': 'mean'
    }).reset_index()
    
    monthly.columns = ['month', 'avg_tip_pct', 'avg_surcharge', 'avg_tip_amount', 'avg_fare']
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[int(x)-1] if x <= 12 else 'Dec')
    
    print("\nMonthly Tip and Surcharge Analysis:")
    print("-" * 60)
    for _, row in monthly.iterrows():
        print(f"  {row['month_name']}: Avg Tip%={row['avg_tip_pct']:.1f}%, Avg Surcharge=${row['avg_surcharge']:.2f}, Avg Tip=${row['avg_tip_amount']:.2f}")
    
    # Calculate correlation
    correlation = monthly['avg_surcharge'].corr(monthly['avg_tip_pct'])
    print(f"\nCorrelation (Surcharge vs Tip %): {correlation:.3f}")
    
    if correlation < -0.3:
        print("Interpretation: NEGATIVE correlation - tips may be crowded out by surcharges")
    elif correlation > 0.3:
        print("Interpretation: POSITIVE correlation - tips increase with surcharges")
    else:
        print("Interpretation: WEAK correlation - no clear relationship")
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=monthly['month_name'],
            y=monthly['avg_surcharge'],
            name='Avg Surcharge ($)',
            marker_color='steelblue',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly['month_name'],
            y=monthly['avg_tip_pct'],
            name='Avg Tip %',
            mode='lines+markers',
            line=dict(color='coral', width=3),
            marker=dict(size=10)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'Tip Crowding Out Analysis (Correlation: {correlation:.2f})',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Average Surcharge ($)", secondary_y=False)
    fig.update_yaxes(title_text="Average Tip %", secondary_y=True)
    
    fig.write_html(str(VISUALS_DIR / "tip_crowding_out.html"))
    fig.write_image(str(VISUALS_DIR / "tip_crowding_out.png"), scale=2)
    
    print(f"\nVisualization saved to: outputs/visuals/tip_crowding_out.html")
    
    # Save data
    monthly.to_csv(OUTPUT_DIR / "tip_surcharge_analysis.csv", index=False)
    
    return monthly


# ============================================================================
# PHASE 4: RAIN TAX ANALYSIS
# ============================================================================

def phase4_rain_elasticity(yellow_ddf, weather_df):
    """
    Phase 4: Rain Elasticity Model
    """
    print("\n" + "=" * 70)
    print("PHASE 4: RAIN ELASTICITY ANALYSIS")
    print("=" * 70)
    
    if weather_df is None:
        print("Error: Weather data not found. Run utils/weather.py first.")
        return None
    
    # Get daily trip counts
    df = yellow_ddf[['tpep_pickup_datetime']].compute()
    df['date'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.date
    daily_counts = df.groupby('date').size().reset_index()
    daily_counts.columns = ['date', 'trip_count']
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # Merge with weather
    merged = daily_counts.merge(weather_df, on='date', how='inner')
    
    print(f"\nData Overview:")
    print(f"  Total days analyzed: {len(merged)}")
    print(f"  Date range: {merged['date'].min().strftime('%Y-%m-%d')} to {merged['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average daily trips: {merged['trip_count'].mean():,.0f}")
    
    # Calculate correlation
    correlation = merged['precipitation_mm'].corr(merged['trip_count'])
    
    # Split by weather condition
    rainy_days = merged[merged['precipitation_mm'] >= 1.0]
    dry_days = merged[merged['precipitation_mm'] < 1.0]
    heavy_rain_days = merged[merged['precipitation_mm'] >= 10.0]
    
    avg_trips_rainy = rainy_days['trip_count'].mean() if len(rainy_days) > 0 else 0
    avg_trips_dry = dry_days['trip_count'].mean() if len(dry_days) > 0 else 0
    avg_trips_heavy = heavy_rain_days['trip_count'].mean() if len(heavy_rain_days) > 0 else 0
    
    pct_change = ((avg_trips_rainy - avg_trips_dry) / avg_trips_dry * 100) if avg_trips_dry > 0 else 0
    pct_change_heavy = ((avg_trips_heavy - avg_trips_dry) / avg_trips_dry * 100) if avg_trips_dry > 0 else 0
    
    print(f"\nWeather Impact Analysis:")
    print(f"  Dry days (<1mm): {len(dry_days)} days, Avg trips: {avg_trips_dry:,.0f}")
    print(f"  Rainy days (>=1mm): {len(rainy_days)} days, Avg trips: {avg_trips_rainy:,.0f} ({pct_change:+.1f}%)")
    print(f"  Heavy rain (>=10mm): {len(heavy_rain_days)} days, Avg trips: {avg_trips_heavy:,.0f} ({pct_change_heavy:+.1f}%)")
    
    print(f"\nElasticity Analysis:")
    print(f"  Correlation coefficient: {correlation:.3f}")
    
    if correlation < -0.3:
        elasticity = "ELASTIC - Demand decreases significantly with rain"
    elif correlation < -0.1:
        elasticity = "MODERATELY ELASTIC - Demand decreases somewhat with rain"
    elif correlation > 0.1:
        elasticity = "INVERSE ELASTIC - Demand INCREASES with rain (people avoid walking)"
    else:
        elasticity = "INELASTIC - Demand is insensitive to rain"
    
    print(f"  Interpretation: {elasticity}")
    
    # Find wettest month
    merged['month'] = merged['date'].dt.month
    monthly_precip = merged.groupby('month')['precipitation_mm'].sum()
    wettest_month = monthly_precip.idxmax()
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    print(f"\nWettest Month: {month_names[wettest_month-1]} ({monthly_precip[wettest_month]:.1f}mm total)")
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged['precipitation_mm'],
        y=merged['trip_count'],
        mode='markers',
        name='Daily Trips',
        marker=dict(size=8, color='steelblue', opacity=0.6),
        hovertemplate='Precipitation: %{x:.1f}mm<br>Trips: %{y:,}<extra></extra>'
    ))
    
    # Add trendline
    x = merged['precipitation_mm'].values
    y = merged['trip_count'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name=f'Trend (r={correlation:.2f})',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=f'Taxi Demand vs Precipitation (Correlation: {correlation:.2f})',
        xaxis_title='Daily Precipitation (mm)',
        yaxis_title='Daily Trip Count',
        height=500,
        showlegend=True
    )
    
    fig.write_html(str(VISUALS_DIR / "rain_elasticity.html"))
    fig.write_image(str(VISUALS_DIR / "rain_elasticity.png"), scale=2)
    
    print(f"\nVisualization saved to: outputs/visuals/rain_elasticity.html")
    
    # Save results
    results = {
        'correlation': float(correlation),
        'elasticity_interpretation': elasticity,
        'avg_trips_dry_days': float(avg_trips_dry),
        'avg_trips_rainy_days': float(avg_trips_rainy),
        'pct_change_rainy': float(pct_change),
        'num_dry_days': int(len(dry_days)),
        'num_rainy_days': int(len(rainy_days)),
        'wettest_month': month_names[wettest_month-1],
        'wettest_month_precipitation_mm': float(monthly_precip[wettest_month])
    }
    
    with open(OUTPUT_DIR / "rain_elasticity_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    merged.to_csv(OUTPUT_DIR / "weather_trip_correlation.csv", index=False)
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis for Phase 2, 3, and 4."""
    start_time = datetime.now()
    
    # Load data
    print("\n[1/6] Loading taxi data...")
    yellow_ddf = load_yellow_taxi()
    green_ddf = load_green_taxi()
    weather_df = load_weather_data()
    
    # Phase 2: Zone-Based Analysis
    print("\n[2/6] Running Phase 2: Zone-Based Analysis...")
    leakage_results = phase2_leakage_audit(yellow_ddf)
    comparison_results = phase2_yellow_vs_green(yellow_ddf, green_ddf)
    
    # Phase 3: Visual Audit
    print("\n[3/6] Running Phase 3a: Border Effect Analysis...")
    border_results = phase3_border_effect(yellow_ddf)
    
    print("\n[4/6] Running Phase 3b: Velocity Heatmap...")
    velocity_results = phase3_velocity_heatmap(yellow_ddf)
    
    print("\n[5/6] Running Phase 3c: Tip Crowding Out Analysis...")
    tip_results = phase3_tip_crowding_out(yellow_ddf)
    
    # Phase 4: Rain Tax Analysis
    print("\n[6/6] Running Phase 4: Rain Elasticity Analysis...")
    rain_results = phase4_rain_elasticity(yellow_ddf, weather_df)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total execution time: {duration:.1f} minutes")
    print(f"\nOutput files generated:")
    print(f"  - outputs/leakage_audit_results.json")
    print(f"  - outputs/yellow_green_comparison.csv")
    print(f"  - outputs/border_effect_data.csv")
    print(f"  - outputs/velocity_data.csv")
    print(f"  - outputs/tip_surcharge_analysis.csv")
    print(f"  - outputs/rain_elasticity_results.json")
    print(f"  - outputs/weather_trip_correlation.csv")
    print(f"\nVisualizations saved to: outputs/visuals/")
    for f in VISUALS_DIR.glob("*"):
        print(f"  - {f.name}")
    
    print("\nAll phases completed successfully!")
    print("Run 'streamlit run outputs/dashboard.py' to view the interactive dashboard.")


if __name__ == "__main__":
    main()
