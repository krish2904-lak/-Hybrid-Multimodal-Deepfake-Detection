"""
Dashboard Page: Analytics and Performance Metrics
Visualizes training convergence, validation metrics, confusion matrix, and ROC-AUC curves.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def render_analytics():
    st.markdown('<div class="page-title">Model Analytics & Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Evaluation benchmarks, convergence trajectories, and statistical diagnostics.</div>',
        unsafe_allow_html=True
    )

    history_file = "outputs/training_history.json"
    val_file = "outputs/validation_metrics.json"

    if not os.path.exists(history_file):
        st.warning(
            "⚠️ No stored training history found in `outputs/training_history.json`. "
            "Run model training via `python src/models/train.py` to generate live metrics."
        )
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    epochs = history.get("epochs", [1])
    train_loss = history.get("train_loss", [0.65])
    val_loss = history.get("val_loss", [0.69])
    train_acc = history.get("train_acc", [0.55])
    val_acc = history.get("val_acc", [0.50])

    # Summary metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Acc", f"{train_acc[-1] * 100:.1f}%")
    with col2:
        st.metric("Final Val Acc", f"{val_acc[-1] * 100:.1f}%")
    with col3:
        st.metric("Final Train Loss", f"{train_loss[-1]:.4f}")
    with col4:
        st.metric("Final Val Loss", f"{val_loss[-1]:.4f}")

    st.markdown("---")

    # Convergence Curves
    st.subheader("1. Training and Validation Convergence")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.write("**Cross-Entropy Loss Trajectory:**")
        loss_df = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss": train_loss,
            "Validation Loss": val_loss
        }).set_index("Epoch")
        st.line_chart(loss_df, color=["#3b82f6", "#ef4444"])

    with c_col2:
        st.write("**Classification Accuracy Trajectory:**")
        acc_df = pd.DataFrame({
            "Epoch": epochs,
            "Train Accuracy": [a * 100 for a in train_acc],
            "Validation Accuracy": [a * 100 for a in val_acc]
        }).set_index("Epoch")
        st.line_chart(acc_df, color=["#10b981", "#8b5cf6"])

    st.markdown("---")

    # Confusion Matrix and ROC Curve
    st.subheader("2. Validation Matrix and ROC-AUC Analysis")
    m_col1, m_col2 = st.columns(2)

    val_data = {}
    if os.path.exists(val_file):
        with open(val_file, "r") as f:
            val_data = json.load(f)

    cm = val_data.get("confusion_matrix", [[8, 0], [8, 0]])
    precision = val_data.get("precision", 0.0)
    recall = val_data.get("recall", 0.0)
    f1 = val_data.get("f1", 0.0)
    roc_auc = val_data.get("roc_auc", 0.50)
    roc_curve_data = val_data.get("roc_curve", {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    with m_col1:
        st.write("**Confusion Matrix (Normalized Counts):**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 12, "color": "white"}
        )
        ax.set_xlabel("Predicted Label", color="#94a3b8")
        ax.set_ylabel("True Ground Truth", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        st.pyplot(fig)

    with m_col2:
        st.write(f"**ROC Curve (AUC = {roc_auc:.2f}):**")
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#0f172a')
        fpr = roc_curve_data.get("fpr", [0, 1])
        tpr = roc_curve_data.get("tpr", [0, 1])
        ax2.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        ax2.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--")
        ax2.set_xlabel("False Positive Rate", color="#94a3b8")
        ax2.set_ylabel("True Positive Rate", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        st.pyplot(fig2)

    st.markdown("---")

    # Metrics Summary Table
    st.subheader("3. Comprehensive Validation Metrics Table")
    metrics_summary_df = pd.DataFrame([
        {"Metric": "Accuracy", "Score": f"{val_acc[-1] * 100:.2f}%", "Domain": "Overall Accuracy"},
        {"Metric": "Precision (Fake)", "Score": f"{precision * 100:.2f}%", "Domain": "Positive Predictive Value"},
        {"Metric": "Recall (Fake)", "Score": f"{recall * 100:.2f}%", "Domain": "True Positive Detection Rate"},
        {"Metric": "F1-Score", "Score": f"{f1 * 100:.2f}%", "Domain": "Harmonic Precision-Recall Mean"},
        {"Metric": "ROC-AUC", "Score": f"{roc_auc:.4f}", "Domain": "Area Under Receiver Operating Curve"}
    ])
    st.table(metrics_summary_df.set_index("Metric"))
