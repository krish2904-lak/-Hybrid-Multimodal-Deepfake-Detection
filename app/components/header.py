"""
UI Styling and Custom Component Utilities
Injects modern dark navy/charcoal styling with purple & blue accents,
responsive cards, and custom metric badges.
"""

import streamlit as st


def inject_custom_css():
    """Injects bespoke dark-themed CSS for high-end professional appearance."""
    st.markdown("""
    <style>
    /* Dark Navy/Charcoal Palette */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
    }

    /* Modern Card Container */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-subtext {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Prediction Badges */
    .badge-fake {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        border-radius: 9999px;
        padding: 0.35rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-real {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        border-radius: 9999px;
        padding: 0.35rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Section Banner */
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Code and Preformatted blocks */
    code {
        color: #a78bfa !important;
        background-color: #1e293b !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", accent_color: str = "#3b82f6"):
    """Renders a styled metric card."""
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {accent_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)
