"""
End-to-End Integration Tests
Validates full pipeline execution from video input to prediction and PDF/CSV reporting.
"""

import os
import pytest

from src.inference.predictor import DeepfakePredictor
from app.components.report_generator import generate_pdf_report, generate_csv_report
from src.evaluation.metrics import calculate_metrics


def test_predictor_end_to_end(tmp_path):
    video_path = "datasets/raw/sample_fake.mp4"
    if not os.path.exists(video_path):
        pytest.skip("sample_fake.mp4 not present in datasets/raw")

    predictor = DeepfakePredictor(checkpoint_path="checkpoints/best_model.pt", sequence_length=8)
    result = predictor.predict_video(video_path)

    assert result["success"] is True
    assert result["prediction"] in ["REAL", "FAKE"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert result["frames_analyzed"] > 0
    assert result["processing_time"] > 0.0

    # Test PDF report generation
    pdf_out = os.path.join(tmp_path, "test_report.pdf")
    generate_pdf_report(result, pdf_out)
    assert os.path.exists(pdf_out)
    assert os.path.getsize(pdf_out) > 1000

    # Test CSV report generation
    csv_text = generate_csv_report(result)
    assert "FORENSIC" in csv_text
    assert result["prediction"] in csv_text


def test_calculate_metrics():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.2, 0.85, 0.4, 0.3]

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 5 / 6
    assert metrics["sample_count"] == 6
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
