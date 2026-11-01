"""
Model Evaluation Metrics Module
Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC
for Deepfake Detection and Adversarial Robustness Analysis.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from typing import Dict, Any, List, Optional, Tuple


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    y_true: 1D array of ground truth (0 = Real, 1 = Fake)
    y_pred: 1D array of predicted class (0 or 1)
    y_prob: 1D or 2D array of predicted probabilities for positive class (Fake)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Positive class is 1 (Fake)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # ROC AUC calculation if probabilities are provided
    roc_auc = 0.5
    fpr_list: List[float] = [0.0, 1.0]
    tpr_list: List[float] = [0.0, 1.0]

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            scores = y_prob[:, 1]
        else:
            scores = y_prob.ravel()

        # Check if there are at least two distinct classes in y_true
        if len(np.unique(y_true)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = float(auc(fpr, tpr))
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
            except Exception:
                roc_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list
        },
        "sample_count": len(y_true)
    }


def compute_robustness_gap(clean_acc: float, adv_acc: float) -> Dict[str, float]:
    """
    Computes adversarial robustness metrics and accuracy degradation gap.
    """
    gap = clean_acc - adv_acc
    relative_degradation = (gap / clean_acc * 100.0) if clean_acc > 0 else 0.0
    return {
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "robustness_gap": max(0.0, gap),
        "relative_degradation_pct": max(0.0, relative_degradation)
    }
