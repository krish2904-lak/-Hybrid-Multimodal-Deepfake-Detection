"""
Dashboard Page: Overview
Displays system health, high-level operational metrics, and recent forensic evaluations.
"""

import os
import json
import torch
import streamlit as st
from app.components.header import render_metric_card


def render_overview():
    st.markdown('<div class="page-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Hybrid Multimodal Federated Learning Framework for Robust Deepfake Detection under Adversarial Conditions</div>',
        unsafe_allow_html=True
    )

    # Operational metrics
    history_file = "outputs/training_history.json"
    trained_acc = "N/A"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                val_accs = data.get("val_acc", [])
                if val_accs:
                    trained_acc = f"{max(val_accs) * 100:.1f}%"
        except Exception:
            pass

    ckpt_exists = os.path.exists("checkpoints/best_model.pt")
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    # Fetch recent analyses count from session state
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []

    history = st.session_state["analysis_history"]
    total_processed = len(history)
    fakes_count = sum(1 for h in history if h.get("prediction") == "FAKE")
    reals_count = sum(1 for h in history if h.get("prediction") == "REAL")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Videos Analyzed", str(total_processed), "Total sessions run", "#3b82f6")
    with col2:
        render_metric_card("Deepfakes Identified", str(fakes_count), "Manipulated media", "#ef4444")
    with col3:
        render_metric_card("Authentic Real Media", str(reals_count), "Verified pristine", "#10b981")
    with col4:
        render_metric_card("Model Benchmark Acc", trained_acc, f"Device: {device_name}", "#8b5cf6")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("System Architecture Status")
        st.info(
            """
            **Hybrid Multimodal Framework Architecture:**
            - **Visual Branch:** ResNet / MobileNetV2 spatial feature extractor + Bidirectional LSTM with temporal attention pooling.
            - **Acoustic Branch:** Mel-spectrogram sequence modeling via LSTM encoder.
            - **Cross-Modal Fusion:** Concatenated feature representations passed to multimodal classifier with automatic unimodal fallback.
            - **Federated Engine:** Decentralized FedAvg coordinator for multi-node cooperative learning without raw data exposure.
            - **Adversarial Resilience:** Robustness stress-testing via FGSM and multi-step PGD perturbation attacks.
            """
        )

        st.subheader("Model Checkpoint Registry")
        status_color = "🟢" if ckpt_exists else "🟡"
        st.write(f"{status_color} **Active Checkpoint:** `checkpoints/best_model.pt` ({'Loaded & Ready' if ckpt_exists else 'Awaiting Training'})")
        st.write(f"⚙️ **PyTorch Runtime:** `{torch.__version__}` | **Compute Device:** `{device_name}`")

    with col_right:
        st.subheader("Recent Analysis Log")
        if not history:
            st.caption("No video analysis conducted yet in this session. Navigate to 'Deepfake Detection' to inspect a video clip.")
        else:
            for item in reversed(history[-5:]):
                pred = item.get("prediction", "UNKNOWN")
                badge_color = "🔴" if pred == "FAKE" else "🟢"
                st.markdown(
                    f"{badge_color} **{item.get('filename')}** — `{pred}` ({item.get('confidence')}%) | *{item.get('timestamp')}*"
                )
