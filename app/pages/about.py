"""
Dashboard Page: About Project
Presents academic abstract, system design, theoretical foundations, and tech stack details.
"""

import streamlit as st


def render_about():
    st.markdown('<div class="page-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Academic project documentation, architectural blueprints, and mathematical foundations.</div>',
        unsafe_allow_html=True
    )

    st.subheader("1. Academic Abstract")
    st.markdown("""
    The rapid proliferation of hyper-realistic generative artificial intelligence, notably deep generative adversarial
    networks (GANs) and diffusion models, presents severe threats to digital trust, biometric identity systems, and
    multimedia forensics. Conventional deepfake detectors depend primarily on single-modality video analysis and centralized
    training pipelines, rendering them vulnerable to privacy violations, cross-dataset distribution shifts, and intentional
    adversarial noise attacks.

    This project develops a **Hybrid Multimodal Federated Learning Framework for Robust Deepfake Detection under Adversarial Conditions**.
    Our system combines:
    1. **Visual Facial Feature Extraction:** Deep CNN backbone (ResNet / MobileNetV2) paired with Bidirectional Long Short-Term Memory (BiLSTM) and Temporal Attention Pooling.
    2. **Acoustic Speech Forensics:** Mel-spectrogram frequency-time decomposition encoded by an acoustic LSTM module.
    3. **Cross-Modal Fusion:** Dense multimodal integration with automatic unimodal visual fallback.
    4. **Federated Learning via FedAvg:** Multi-client collaborative training preserving user privacy without exposing raw media.
    5. **Adversarial Hardening:** Robustness validation against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).
    """)

    st.markdown("---")

    st.subheader("2. System Architecture Flow")
    st.code("""
VIDEO INPUT (.mp4, .avi, .mov)
     │
     ├───> AUDIO EXTRACTION ───> Log-Mel Spectrogram ───> Acoustic LSTM ──────────┐
     │                                                                             │
     └───> FRAME EXTRACTION ───> Face Detection ───> Visual CNN ───> BiLSTM ───> FUSION LAYER ───> VERDICT (REAL / FAKE)
                                                                                   │
                                                      [FEDERATED LEARNING: FedAvg] │
                                                      [ADVERSARIAL STRESS: FGSM/PGD]
    """, language="text")

    st.markdown("---")

    st.subheader("3. Core Mathematical Formulations")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Federated Averaging (FedAvg)")
        st.latex(r"W_{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_t^k")
        st.caption("Central server aggregates client weight updates weighted by client dataset size $n_k$.")

        st.markdown("#### Fast Gradient Sign Method (FGSM)")
        st.latex(r"x_{adv} = \text{clamp}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y)), 0, 1)")
        st.caption("One-step white-box adversarial perturbation with $L_\infty$ bound $\epsilon$.")

    with col_m2:
        st.markdown("#### Temporal Attention Weighting")
        st.latex(r"\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^T \exp(e_j)}, \quad v = \sum_{t=1}^T \alpha_t h_t")
        st.caption("Focuses temporal weighting on frames displaying high manipulation artifacts.")

        st.markdown("#### Projected Gradient Descent (PGD)")
        st.latex(r"x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sign}(\nabla_{x^t} \mathcal{L}) \right)")
        st.caption("Multi-step iterative projection within the $L_\infty$ epsilon ball.")

    st.markdown("---")

    st.subheader("4. Technology Stack")
    st.markdown("""
    - **Deep Learning Framework:** PyTorch 2.x, Torchvision, Torchaudio
    - **Computer Vision & Signal Processing:** OpenCV, Librosa, SoundFile, PIL, NumPy
    - **Federated Simulation & Machine Learning:** Scikit-Learn, SciPy, FedAvg Algorithm
    - **Dashboard UI & Visualization:** Streamlit, Matplotlib, Seaborn, Altair
    - **Forensic Reporting:** ReportLab PDF Engine, Python CSV Engine
    - **Testing & Quality Assurance:** Pytest Test Suite
    """)

    st.markdown("---")

    st.subheader("5. Academic Project Credits")
    st.markdown("""
    - **Project Title:** Hybrid Multimodal Federated Learning Framework for Robust Deepfake Detection under Adversarial Conditions
    - **Domain:** Artificial Intelligence, Deep Learning, Multimedia Forensics & Cybersecurity
    - **Academic Year:** B.Tech Final Year Capstone Project
    """)
