"""
Main Streamlit Dashboard Entrypoint
Integrates all 6 forensic pages with a unified dark navy UI and responsive sidebar navigation.
"""

import sys
import os

# Add root directory to path for clean package imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st
from app.components.header import inject_custom_css
from app.pages.overview import render_overview
from app.pages.detect import render_detect
from app.pages.analytics import render_analytics
from app.pages.federated_learning import render_federated_learning
from app.pages.adversarial_testing import render_adversarial_testing
from app.pages.about import render_about


def main():
    st.set_page_config(
        page_title="Deepfake Forensic AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply Dark Navy Theme styling
    inject_custom_css()

    # Sidebar Navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h2 style="color: #38bdf8; margin: 0; font-size: 1.4rem;">🛡️ DEEPFAKE AI</h2>
        <p style="color: #94a3b8; font-size: 0.78rem; margin: 0;">Multimodal Federated Forensics</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation Menu:",
        [
            "📊 Overview",
            "🔍 Deepfake Detection",
            "📈 Analytics & Metrics",
            "🌐 Federated Learning",
            "🛡️ Adversarial Robustness",
            "ℹ️ About Project"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.78rem; color: #64748b; padding: 0.5rem;">
        <b>B.Tech Final Year Capstone</b><br>
        Framework: PyTorch + Streamlit<br>
        Status: <span style="color: #34d399;">● Online</span>
    </div>
    """, unsafe_allow_html=True)

    # Route to selected page
    if page == "📊 Overview":
        render_overview()
    elif page == "🔍 Deepfake Detection":
        render_detect()
    elif page == "📈 Analytics & Metrics":
        render_analytics()
    elif page == "🌐 Federated Learning":
        render_federated_learning()
    elif page == "🛡️ Adversarial Robustness":
        render_adversarial_testing()
    elif page == "ℹ️ About Project":
        render_about()


if __name__ == "__main__":
    main()
