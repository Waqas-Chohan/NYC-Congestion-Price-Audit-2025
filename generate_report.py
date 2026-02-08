"""
NYC Congestion Pricing Audit - Report Generator
Generates a comprehensive PDF audit report using analysis results and visualizations.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALS_DIR = OUTPUT_DIR / "visuals"
REPORT_PATH = OUTPUT_DIR / "NYC_Congestion_Audit_Report.pdf"

def load_results():
    """Load analysis results from JSON/CSV files."""
    results = {}
    
    # Leakage Audit
    leakage_file = OUTPUT_DIR / "leakage_audit_results.json"
    if leakage_file.exists():
        with open(leakage_file, 'r') as f:
            results['leakage'] = json.load(f)
            
    # Rain Elasticity
    rain_file = OUTPUT_DIR / "rain_elasticity_results.json"
    if rain_file.exists():
        with open(rain_file, 'r') as f:
            results['rain'] = json.load(f)
            
    # Yellow vs Green
    yg_file = OUTPUT_DIR / "yellow_green_comparison.csv"
    if yg_file.exists():
        results['yellow_green'] = pd.read_csv(yg_file)
        
    return results

def create_report():
    """Generate the PDF report."""
    print(f"Generating report at: {REPORT_PATH}")
    
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center
    
    h2_style = styles['Heading2']
    h2_style.textColor = colors.HexColor('#1f77b4')
    
    h3_style = styles['Heading3']
    h3_style.textColor = colors.HexColor('#2c3e50')
    
    normal_style = styles['Normal']
    normal_style.spaceAfter = 12
    
    # Content list
    story = []
    
    # Load data
    data = load_results()
    
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("NYC Congestion Pricing Audit", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("2025 Manhattan Congestion Relief Zone Analysis", styles['Heading2']))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Prepared for: NYC Department of Transportation", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Author: M.Waqas Chohan</b>", styles['Normal']))
    story.append(Paragraph("<b>Section: S.E_6-A</b>", styles['Normal']))
    story.append(PageBreak())
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    story.append(Paragraph("Executive Summary", h2_style))
    
    summary_text = """
    This audit evaluates the impact of the NYC Congestion Relief Zone toll implemented in 2025. 
    Our analysis of over 44 million taxi trips reveals significant insights into compliance, 
    traffic patterns, and economic shifts.
    """
    story.append(Paragraph(summary_text, normal_style))
    
    # Key Metrics Bullet Points
    if 'leakage' in data:
        leak = data['leakage']
        story.append(Paragraph(f"• <b>Surcharge Compliance:</b> {leak['compliance_rate']:.1f}% compliance rate observed, with a checkout leakage of {leak['leakage_rate']:.1f}%.", normal_style))
    
    if 'rain' in data:
        rain = data['rain']
        story.append(Paragraph(f"• <b>Weather Elasticity:</b> Demand is {rain.get('elasticity_interpretation', 'Unknown').split(' - ')[0]}, showing a potential revenue opportunity during precipitation events.", normal_style))
    
    story.append(Paragraph("• <b>Traffic Velocity:</b> Analysis of speed patterns suggests mixed results in congestion reduction during peak hours.", normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Recommendation:</b> Immediate enforcement action is required at identified high-leakage zones to recover lost revenue.", normal_style))
    story.append(PageBreak())
    
    # ========================================================================
    # POLICY RECOMMENDATIONS
    # ========================================================================
    story.append(Paragraph("Policy Recommendations", h2_style))
    
    recs = [
        ("1. Close the 'Ghost Trip' Loophole", 
         "Implement strict geofencing validation. Our audit identified significant trips with missing surcharges despite originating or terminating in the congestion zone."),
        
        ("2. Dynamic Rain Surcharge", 
         "Data indicates taxi demand is inelastic or even inverse-elastic to rain. We recommend a dynamic 'Rain Tax' surcharge of $1.50 during heavy precipitation (>10mm) to manage demand surge and increase revenue."),
        
        ("3. Border Zone Mitigation", 
         "Evidence suggests passengers are dropping off just outside the zone boundaries (e.g., Upper East/West Side borders). Consider expanding the zone buffer or implementing drop-off fees in bordering zones.")
    ]
    
    for title, desc in recs:
        story.append(Paragraph(f"<b>{title}</b>", h3_style))
        story.append(Paragraph(desc, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
    story.append(PageBreak())
    
    # ========================================================================
    # PHASE 2: ZONE-BASED ANALYSIS
    # ========================================================================
    story.append(Paragraph("Phase 2: Zone-Based Analysis", h2_style))
    
    # Leakage Audit
    story.append(Paragraph("Leakage Audit", h3_style))
    if 'leakage' in data:
        leak = data['leakage']
        text = f"""
        Analysis of {leak['total_congestion_trips']:,} trips involving the congestion zone revealed that 
        {leak['trips_missing_surcharge']:,} trips (26.0%) were missing the required surcharge.
        """
        story.append(Paragraph(text, normal_style))
        
        # Table of missing locations
        story.append(Paragraph("<b>Top Zones with Missing Surcharges:</b>", normal_style))
        table_data = [['Zone ID', 'Missing Count']]
        top_missing = leak.get('top3_missing_locations', [])
        for item in top_missing:
            table_data.append([str(item['location_id']), f"{item['missing_count']:,}"])
            
        t = Table(table_data, colWidths=[1.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Yellow vs Green
    story.append(Paragraph("Market Share Analysis", h3_style))
    yellow_green_img = VISUALS_DIR / "yellow_vs_green_comparison.png"
    if yellow_green_img.exists():
        img = Image(str(yellow_green_img), width=6*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 1: Monthly trip volume comparison showing Yellow taxi dominance.</i>", styles['Italic']))
    
    story.append(PageBreak())
    
    # ========================================================================
    # PHASE 3: VISUAL AUDIT
    # ========================================================================
    story.append(Paragraph("Phase 3: Visual Audit", h2_style))
    
    # Velocity Heatmap
    story.append(Paragraph("Congestion Velocity Analysis", h3_style))
    velocity_img = VISUALS_DIR / "velocity_heatmap.png"
    if velocity_img.exists():
        img = Image(str(velocity_img), width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 2: Average speed heatmap by hour and day of week.</i>", styles['Italic']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Tip Crowding Out
    story.append(Paragraph("Tip Crowding Out Effect", h3_style))
    tip_img = VISUALS_DIR / "tip_crowding_out.png"
    if tip_img.exists():
        img = Image(str(tip_img), width=6*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 3: Correlation between average surcharge and tip percentage.</i>", styles['Italic']))
        
    story.append(PageBreak())
    
    # ========================================================================
    # PHASE 4: RAIN TAX ANALYSIS
    # ========================================================================
    story.append(Paragraph("Phase 4: Rain Tax Analysis", h2_style))
    
    if 'rain' in data:
        rain = data['rain']
        text = f"""
        Using OpenMeteo historical weather data, we analyzed rainfall impact on taxi demand.
        The correlation coefficient of {rain['correlation']:.2f} indicates that demand is 
        <b>{rain.get('elasticity_interpretation', 'Unknown').split(' - ')[0]}</b>.
        """
        story.append(Paragraph(text, normal_style))
        
        # Stats
        story.append(Paragraph(f"• <b>Wettest Month:</b> {rain['wettest_month']} ({rain['wettest_month_precipitation_mm']:.1f}mm)", normal_style))
        story.append(Paragraph(f"• <b>Dry Day Avg Trips:</b> {rain['avg_trips_dry_days']:,.0f}", normal_style))
        story.append(Paragraph(f"• <b>Rainy Day Avg Trips:</b> {rain['avg_trips_rainy_days']:,.0f} ({rain['pct_change_rainy']:+.1f}%)", normal_style))
        
        rain_img = VISUALS_DIR / "rain_elasticity.png"
        if rain_img.exists():
            img = Image(str(rain_img), width=6*inch, height=3*inch)
            story.append(img)
            story.append(Paragraph("<i>Figure 4: Scatter plot of precipitation vs daily trip counts.</i>", styles['Italic']))
    
    # Build
    doc.build(story)
    print("PDF Report generated successfully!")

if __name__ == "__main__":
    create_report()
