"""
Analysis Report Generator Module
Generates downloadable PDF and CSV forensic analysis reports for evaluated videos.
"""

import os
import csv
import io
from datetime import datetime
from typing import Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def generate_csv_report(analysis_result: Dict[str, Any]) -> str:
    """Generates CSV format string of the forensic analysis."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["HYBRID MULTIMODAL DEEPFAKE DETECTION FORENSIC REPORT"])
    writer.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["PARAMETER", "VALUE"])
    writer.writerow(["Filename", analysis_result.get("filename", "N/A")])
    writer.writerow(["Final Prediction", analysis_result.get("prediction", "N/A")])
    writer.writerow(["Confidence (%)", analysis_result.get("confidence", "N/A")])
    writer.writerow(["Real Probability (%)", analysis_result.get("real_probability", "N/A")])
    writer.writerow(["Fake Probability (%)", analysis_result.get("fake_probability", "N/A")])
    writer.writerow(["Frames Analyzed", analysis_result.get("frames_analyzed", "N/A")])
    writer.writerow(["Faces Detected", analysis_result.get("faces_detected", "N/A")])
    writer.writerow(["Face Detection Rate (%)", analysis_result.get("detection_rate", "N/A")])
    writer.writerow(["Modality Active", analysis_result.get("modality_used", "N/A")])
    writer.writerow(["Audio Stream Present", analysis_result.get("has_audio", False)])
    writer.writerow(["Processing Time (s)", analysis_result.get("processing_time", "N/A")])
    writer.writerow([])
    writer.writerow(["FRAME-LEVEL FORENSIC BREAKDOWN"])
    writer.writerow(["Frame Index", "Face Detected", "Fake Anomaly Score (%)", "Prediction"])

    for fr in analysis_result.get("frame_results", []):
        writer.writerow([
            fr.get("frame_index"),
            "Yes" if fr.get("face_detected") else "No",
            fr.get("fake_score"),
            fr.get("prediction")
        ])

    return output.getvalue()


def generate_pdf_report(analysis_result: Dict[str, Any], output_path: str) -> str:
    """Generates a professional forensic PDF report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
        alignment=1
    )
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )

    story = []

    # Header
    story.append(Paragraph("DEEPFAKE FORENSIC ANALYSIS REPORT", title_style))
    story.append(Paragraph(
        "Hybrid Multimodal Federated Learning Framework for Robust Deepfake Detection",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=15))

    # Summary Table
    pred = analysis_result.get("prediction", "UNKNOWN")
    pred_color = colors.HexColor("#ef4444") if pred == "FAKE" else colors.HexColor("#10b981")

    summary_data = [
        ["Target Video File", analysis_result.get("filename", "N/A"), "Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Final Verdict", Paragraph(f"<b><font color='{pred_color.hexval()}'>{pred}</font></b>", styles['Normal']),
         "Confidence", f"{analysis_result.get('confidence', 0.0)}%"],
        ["Frames Processed", str(analysis_result.get("frames_analyzed", 0)),
         "Faces Detected", f"{analysis_result.get('faces_detected', 0)} ({analysis_result.get('detection_rate', 0)}%)"],
        ["Modality Pipeline", analysis_result.get("modality_used", "N/A"),
         "Audio Track", "Active" if analysis_result.get("has_audio") else "Silent / Absent"],
        ["Inference Latency", f"{analysis_result.get('processing_time', 0.0)} s",
         "Model Architecture", "Hybrid Multimodal BiLSTM + ResNet"]
    ]

    summary_table = Table(summary_data, colWidths=[130, 140, 120, 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Frame-Level Table
    story.append(Paragraph("Frame-Level Anomaly Breakdown", section_style))
    frame_table_data = [["Frame #", "Face Detected", "Fake Probability", "Predicted Class"]]

    for fr in analysis_result.get("frame_results", [])[:15]:
        frame_table_data.append([
            str(fr.get("frame_index")),
            "Yes" if fr.get("face_detected") else "No",
            f"{fr.get('fake_score')}%",
            fr.get("prediction")
        ])

    frame_table = Table(frame_table_data, colWidths=[90, 130, 160, 160])
    frame_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(frame_table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=10))
    story.append(Paragraph(
        "Confidential Academic & Forensic Assessment Report. Verified with Hybrid Multimodal Deep Learning.",
        subtitle_style
    ))

    doc.build(story)
    return output_path
