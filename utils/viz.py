"""
Visualization Module
Plotting helpers for maps, heatmaps, and charts
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins

# Add parent directory to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    VISUALS_DIR, COLORMAP_HEATMAP, COLORMAP_CHOROPLETH,
    HOURS_OF_DAY, DAYS_OF_WEEK
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100


def create_border_effect_choropleth(zone_data: pd.DataFrame,
                                    zone_gdf: gpd.GeoDataFrame,
                                    value_col: str = 'pct_change',
                                    zone_id_col: str = 'zone_id',
                                    title: str = "Border Effect: % Change in Drop-offs (2024 vs 2025)",
                                    output_path: str = None) -> str:
    """
    Create choropleth map showing border effect.
    
    Args:
        zone_data: DataFrame with zone statistics
        zone_gdf: GeoDataFrame with zone geometries
        value_col: Column with values to plot
        zone_id_col: Column with zone IDs
        title: Map title
        output_path: Path to save HTML file
        
    Returns:
        Path to saved map
    """
    logger.info("Creating border effect choropleth map...")
    
    # Merge data with geometries
    map_data = zone_gdf.merge(
        zone_data,
        left_on='LocationID',
        right_on=zone_id_col,
        how='left'
    )
    
    # Create folium map centered on Manhattan
    m = folium.Map(
        location=[40.7580, -73.9855],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Create choropleth
    folium.Choropleth(
        geo_data=map_data,
        name='choropleth',
        data=zone_data,
        columns=[zone_id_col, value_col],
        key_on='feature.properties.LocationID',
        fill_color='RdYlBu_r',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color='lightgray'
    ).add_to(m)
    
    # Add tooltips
    folium.GeoJson(
        map_data,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['zone', 'Borough', value_col],
            aliases=['Zone:', 'Borough:', '% Change:'],
            localize=True
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    if output_path is None:
        output_path = VISUALS_DIR / "border_effect_map.html"
    
    m.save(str(output_path))
    logger.info(f"✓ Choropleth map saved to: {output_path}")
    
    return str(output_path)


def create_velocity_heatmap(speed_data: pd.DataFrame,
                            hour_col: str = 'hour',
                            day_col: str = 'day_of_week',
                            speed_col: str = 'avg_speed_mph',
                            title: str = "Average Trip Speed (mph)",
                            output_path: str = None) -> str:
    """
    Create heatmap of average speed by hour and day of week.
    
    Args:
        speed_data: DataFrame with aggregated speed data
        hour_col: Column with hour of day (0-23)
        day_col: Column with day of week (0-6)
        speed_col: Column with average speed
        title: Plot title
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating velocity heatmap...")
    
    # Pivot data for heatmap
    heatmap_data = speed_data.pivot(
        index=day_col,
        columns=hour_col,
        values=speed_col
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap=COLORMAP_HEATMAP,
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Average Speed (mph)'},
        ax=ax
    )
    
    # Customize
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    
    # Day labels
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax.set_yticklabels(day_labels, rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = VISUALS_DIR / f"velocity_heatmap_{title.replace(' ', '_')}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Velocity heatmap saved to: {output_path}")
    
    return str(output_path)


def create_dual_axis_chart(monthly_data: pd.DataFrame,
                          date_col: str = 'month',
                          bar_col: str = 'avg_surcharge',
                          line_col: str = 'avg_tip_pct',
                          bar_label: str = 'Average Surcharge ($)',
                          line_label: str = 'Average Tip %',
                          title: str = "Tip Crowding Out Analysis",
                          output_path: str = None) -> str:
    """
    Create dual-axis chart with bar and line.
    
    Args:
        monthly_data: DataFrame with monthly aggregates
        date_col: Column with dates/months
        bar_col: Column for bar chart (left axis)
        line_col: Column for line chart (right axis)
        bar_label: Label for bar axis
        line_label: Label for line axis
        title: Chart title
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating dual-axis chart...")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart (surcharge)
    fig.add_trace(
        go.Bar(
            x=monthly_data[date_col],
            y=monthly_data[bar_col],
            name=bar_label,
            marker_color='steelblue',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Add line chart (tip percentage)
    fig.add_trace(
        go.Scatter(
            x=monthly_data[date_col],
            y=monthly_data[line_col],
            name=line_label,
            mode='lines+markers',
            line=dict(color='coral', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        title_font_size=16,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        width=1000
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text=bar_label, secondary_y=False)
    fig.update_yaxes(title_text=line_label, secondary_y=True)
    
    # Save figure
    if output_path is None:
        output_path = VISUALS_DIR / "tip_crowding_out.html"
    
    fig.write_html(str(output_path))
    logger.info(f"✓ Dual-axis chart saved to: {output_path}")
    
    return str(output_path)


def create_scatter_plot(data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       x_label: str = None,
                       y_label: str = None,
                       title: str = "Scatter Plot",
                       show_regression: bool = True,
                       output_path: str = None) -> str:
    """
    Create scatter plot with optional regression line.
    
    Args:
        data: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        show_regression: If True, add regression line
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating scatter plot...")
    
    # Use plotly for interactive scatter
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        trendline='ols' if show_regression else None,
        title=title,
        labels={x_col: x_label or x_col, y_col: y_label or y_col}
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=900,
        hovermode='closest'
    )
    
    # Save figure
    if output_path is None:
        output_path = VISUALS_DIR / f"scatter_{x_col}_vs_{y_col}.html"
    
    fig.write_html(str(output_path))
    logger.info(f"✓ Scatter plot saved to: {output_path}")
    
    return str(output_path)


def create_comparison_heatmaps(data_2024: pd.DataFrame,
                               data_2025: pd.DataFrame,
                               hour_col: str = 'hour',
                               day_col: str = 'day_of_week',
                               value_col: str = 'avg_speed_mph',
                               output_path: str = None) -> str:
    """
    Create side-by-side heatmaps for before/after comparison.
    
    Args:
        data_2024: DataFrame with 2024 data
        data_2025: DataFrame with 2025 data
        hour_col: Column with hour
        day_col: Column with day of week
        value_col: Column with values
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating comparison heatmaps...")
    
    # Pivot data
    heatmap_2024 = data_2024.pivot(index=day_col, columns=hour_col, values=value_col)
    heatmap_2025 = data_2025.pivot(index=day_col, columns=hour_col, values=value_col)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Determine shared color scale
    vmin = min(heatmap_2024.min().min(), heatmap_2025.min().min())
    vmax = max(heatmap_2024.max().max(), heatmap_2025.max().max())
    
    # Create heatmaps
    sns.heatmap(heatmap_2024, cmap=COLORMAP_HEATMAP, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Average Speed (mph)'}, ax=ax1)
    ax1.set_title('Q1 2024 (Before Congestion Pricing)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Day of Week', fontsize=12)
    
    sns.heatmap(heatmap_2025, cmap=COLORMAP_HEATMAP, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Average Speed (mph)'}, ax=ax2)
    ax2.set_title('Q1 2025 (After Congestion Pricing)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    
    # Day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax1.set_yticklabels(day_labels, rotation=0)
    ax2.set_yticklabels([])
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = VISUALS_DIR / "velocity_comparison_heatmaps.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Comparison heatmaps saved to: {output_path}")
    
    return str(output_path)


def create_bar_chart(data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    title: str,
                    x_label: str = None,
                    y_label: str = None,
                    color: str = 'steelblue',
                    output_path: str = None) -> str:
    """
    Create simple bar chart.
    
    Args:
        data: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        x_label: Label for x-axis
        y_label: Label for y-axis
        color: Bar color
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    logger.info(f"Creating bar chart: {title}")
    
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_label or x_col, y_col: y_label or y_col},
        color_discrete_sequence=[color]
    )
    
    fig.update_layout(height=500, width=800)
    
    if output_path is None:
        output_path = VISUALS_DIR / f"bar_{title.replace(' ', '_')}.html"
    
    fig.write_html(str(output_path))
    logger.info(f"✓ Bar chart saved to: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    """
    Test visualization functions with sample data.
    """
    print("Visualization module loaded successfully.")
    print(f"Charts will be saved to: {VISUALS_DIR}")
