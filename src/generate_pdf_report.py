import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from src.config import config

def generate_report():
    report_path = os.path.join(config.paths.reports_dir, "Housing_Price_Prediction_Report.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    story = []
    
    # Title
    story.append(Paragraph("Housing Price Prediction Report", title_style))
    story.append(Spacer(1, 12))
    
    # Introduction
    story.append(Paragraph("This document provides a professional summary of the Ames Housing Price Prediction model, generated via Platypus.", normal_style))
    story.append(Spacer(1, 12))
    
    # Feature Importance
    story.append(Paragraph("1. Feature Importance", heading_style))
    feature_imp_path = os.path.join(config.paths.reports_dir, "feature_importance.png")
    if os.path.exists(feature_imp_path):
        story.append(Image(feature_imp_path, width=400, height=300))
    else:
        story.append(Paragraph("Feature importance plot missing.", normal_style))
    story.append(Spacer(1, 12))
    
    # Residual Plot
    story.append(Paragraph("2. Residual Plot", heading_style))
    res_path = os.path.join(config.paths.reports_dir, "residual_plot.png")
    if os.path.exists(res_path):
        story.append(Image(res_path, width=400, height=240))
    else:
        story.append(Paragraph("Residual plot missing.", normal_style))
    story.append(Spacer(1, 12))
    
    # Heatmap
    story.append(Paragraph("3. Correlation Heatmap", heading_style))
    heat_path = os.path.join(config.paths.reports_dir, "correlation_heatmap.png")
    if os.path.exists(heat_path):
        story.append(Image(heat_path, width=400, height=320))
    else:
        story.append(Paragraph("Heatmap missing.", normal_style))
        
    doc.build(story)
    print(f"Professional PDF report successfully generated at {report_path}")

if __name__ == "__main__":
    generate_report()
