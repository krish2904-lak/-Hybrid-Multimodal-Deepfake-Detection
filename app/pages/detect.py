"""
Dashboard Page: Deepfake Detection
Interactive end-to-end video analysis pipeline with frame-level breakdown and report downloads.
"""

import os
import tempfile
from datetime import datetime
import pandas as pd
import streamlit as st

from src.inference.predictor import DeepfakePredictor
from app.components.report_generator import generate_pdf_report, generate_csv_report


@st.cache_resource
def get_cached_predictor():
    """Initializes and caches model predictor instance."""
    return DeepfakePredictor(checkpoint_path="checkpoints/best_model.pt")


def render_detect():
    st.markdown('<div class="page-title">Deepfake Video Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload a video file for spatial, temporal, and multimodal acoustic deepfake analysis.</div>',
        unsafe_allow_html=True
    )

    predictor = get_cached_predictor()

    col_up, col_preview = st.columns([1, 1])

    with col_up:
        st.subheader("1. Video Source")
        upload_mode = st.radio("Select Input Method:", ["Upload Video File", "Use Sample Video"], horizontal=True)

        selected_video_path = None
        video_filename = None

        if upload_mode == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Choose a video file (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov", "webm", "mkv"]
            )
            if uploaded_file is not None:
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_video_path = temp_path
                video_filename = uploaded_file.name
        else:
            # Pre-existing sample video options
            sample_candidates = []
            if os.path.exists("datasets/raw/sample_fake.mp4"):
                sample_candidates.append("datasets/raw/sample_fake.mp4")
            for root, _, files in os.walk("datasets"):
                for f in files:
                    if f.endswith((".mp4", ".avi")):
                        sample_candidates.append(os.path.join(root, f))

            if sample_candidates:
                selected_sample = st.selectbox("Select demo clip:", sample_candidates)
                selected_video_path = selected_sample
                video_filename = os.path.basename(selected_sample)
            else:
                st.info("No sample videos found in `datasets/raw/`. Please upload a video file above.")

        seq_len = st.slider("Analysis Sequence Length (Frames):", min_value=6, max_value=32, value=16, step=2)

    with col_preview:
        st.subheader("2. Video Preview")
        if selected_video_path and os.path.exists(selected_video_path):
            st.video(selected_video_path)
        else:
            st.info("Upload or select a video to preview playback.")

    st.markdown("---")

    # Run Analysis Button
    if selected_video_path:
        analyze_btn = st.button("🚀 Analyze Video for Deepfake Manipulation", type="primary", use_container_width=True)

        if analyze_btn:
            predictor.sequence_length = seq_len
            progress_bar = st.progress(0, text="Validating video file...")

            try:
                progress_bar.progress(20, text="Extracting video frames and facial boundaries...")
                res = predictor.predict_video(selected_video_path)
                progress_bar.progress(70, text="Executing multimodal deep neural network forward pass...")
                progress_bar.progress(100, text="Analysis complete!")

                if not res["success"]:
                    st.error(f"Analysis failed: {res.get('error')}")
                    return

                # Record to session state history
                if "analysis_history" not in st.session_state:
                    st.session_state["analysis_history"] = []

                res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["analysis_history"].append(res)

                # Display Results Header
                st.subheader("3. Forensic Analysis Verdict")
                v_col1, v_col2 = st.columns([2, 3])

                with v_col1:
                    is_fake = res["prediction"] == "FAKE"
                    badge_html = f'<div class="badge-fake">🚨 MANIPULATED (FAKE)</div>' if is_fake else '<div class="badge-real">✅ AUTHENTIC (REAL)</div>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"### Confidence: **{res['confidence']}%**")
                    st.caption(f"Inference latency: {res['processing_time']}s | Modality: `{res['modality_used']}`")

                with v_col2:
                    st.write("**Authenticity Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Class": ["Authentic Real", "Deepfake Manipulation"],
                        "Probability (%)": [res["real_probability"], res["fake_probability"]]
                    })
                    st.bar_chart(prob_df.set_index("Class"), color="#3b82f6")

                # Frame-level metrics
                st.markdown("---")
                st.subheader("4. Temporal Frame-by-Frame Anomaly Trace")
                st.caption("Temporal inconsistency scores across extracted sequential frames:")

                frame_data = res.get("frame_results", [])
                if frame_data:
                    chart_df = pd.DataFrame({
                        "Frame #": [f["frame_index"] for f in frame_data],
                        "Fake Probability (%)": [f["fake_score"] for f in frame_data]
                    }).set_index("Frame #")
                    st.line_chart(chart_df, color="#ef4444")

                # Face previews
                st.markdown("---")
                st.subheader("5. Extracted Face Previews")
                face_crops = res.get("face_crops_preview", [])
                if len(face_crops) > 0:
                    cols = st.columns(min(6, len(face_crops)))
                    for idx, crop in enumerate(face_crops[:6]):
                        with cols[idx]:
                            st.image(crop, caption=f"Frame {idx+1}", use_container_width=True)
                else:
                    st.caption("No facial bounding box crops to preview.")

                # Downloadable Reports
                st.markdown("---")
                st.subheader("6. Export Forensic Analysis Report")
                d_col1, d_col2 = st.columns(2)

                # Generate reports
                pdf_path = os.path.join(tempfile.gettempdir(), f"Forensic_Report_{res['filename']}.pdf")
                generate_pdf_report(res, pdf_path)

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                csv_text = generate_csv_report(res)

                with d_col1:
                    st.download_button(
                        label="📄 Download Forensic Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"Deepfake_Report_{res['filename']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                with d_col2:
                    st.download_button(
                        label="📊 Download Frame Data (CSV)",
                        data=csv_text,
                        file_name=f"Deepfake_Data_{res['filename']}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Inference error encountered: {str(e)}")
