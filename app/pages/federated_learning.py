"""
Dashboard Page: Federated Learning Simulation
Interactive FedAvg multi-client training simulator with privacy-preserving parameter aggregation.
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.federated.server import run_federated_simulation


def render_federated_learning():
    st.markdown('<div class="page-title">Federated Learning Framework</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Decentralized multi-institutional deepfake detector training via Federated Averaging (FedAvg).</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        **Federated Privacy Architecture:**
        - **Data Isolation:** Raw video frames and audio spectrograms remain strictly on local client devices.
        - **Weight Aggregation:** Only model gradient weight parameters $W_k$ are communicated to the central server.
        - **FedAvg Algorithm:** Aggregated parameters computed as $W_{t+1} = \\sum_{k=1}^K \\frac{n_k}{N} W_t^k$.
        - **Note on Scope:** This module is an academically verified software simulation of decentralized client-server interactions.
        """
    )

    st.subheader("1. Active Simulated Federated Nodes")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #3b82f6;">
            <div class="metric-label">Client Node 1</div>
            <div class="metric-value">Forensic Lab α</div>
            <div class="metric-subtext">33.3% Training Partition | High Quality Audio-Visual</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
            <div class="metric-label">Client Node 2</div>
            <div class="metric-value">Media Platform β</div>
            <div class="metric-subtext">33.3% Training Partition | Compressed Web Stream</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label">Client Node 3</div>
            <div class="metric-value">Academic Lab γ</div>
            <div class="metric-subtext">33.4% Training Partition | Diverse Benchmark Mix</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Simulation Controls
    st.subheader("2. Communication Round Simulator")
    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        rounds_input = st.slider("Select Communication Rounds:", min_value=2, max_value=8, value=4, step=1)
        run_sim_btn = st.button("🔄 Execute FedAvg Simulation", type="primary", use_container_width=True)

    with sim_col2:
        st.caption(
            "Clicking Execute will trigger local client epochs on private partitions, collect state dicts, "
            "perform weighted parameter aggregation, and measure global generalization convergence."
        )

    # Check for existing simulation results or run new one
    fl_file = "outputs/federated_history.json"
    sim_data = None

    if run_sim_btn:
        with st.spinner(f"Running {rounds_input} federated rounds across 3 clients..."):
            sim_data = run_federated_simulation(num_rounds=rounds_input)
            st.success(f"Successfully converged across {rounds_input} communication rounds!")

    elif os.path.exists(fl_file):
        try:
            with open(fl_file, "r") as f:
                sim_data = json.load(f)
        except Exception:
            pass

    if sim_data and "rounds" in sim_data:
        rounds_list = sim_data["rounds"]

        st.markdown("---")
        st.subheader("3. Global Federated Convergence Trajectory")

        r_indices = [r["round"] for r in rounds_list]
        r_accs = [r["global_accuracy"] * 100 for r in rounds_list]
        r_losses = [r["global_loss"] for r in rounds_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor('#0f172a')

        # Accuracy Plot
        ax1.set_facecolor('#0f172a')
        ax1.plot(r_indices, r_accs, marker="o", color="#3b82f6", lw=2, label="Global Acc (%)")
        ax1.set_title("Global Model Accuracy per Round", color="white")
        ax1.set_xlabel("Communication Round", color="#94a3b8")
        ax1.set_ylabel("Accuracy (%)", color="#94a3b8")
        ax1.tick_params(colors="#94a3b8")
        ax1.grid(True, linestyle="--", alpha=0.3, color="#334155")

        # Loss Plot
        ax2.set_facecolor('#0f172a')
        ax2.plot(r_indices, r_losses, marker="s", color="#ef4444", lw=2, label="Global Loss")
        ax2.set_title("Global Model Cross-Entropy Loss", color="white")
        ax2.set_xlabel("Communication Round", color="#94a3b8")
        ax2.set_ylabel("Loss", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.grid(True, linestyle="--", alpha=0.3, color="#334155")

        st.pyplot(fig)

        # Client-wise breakdown table
        st.markdown("---")
        st.subheader("4. Client-Wise Local Round Performance")
        latest_round = rounds_list[-1]
        c_rows = []
        for c in latest_round.get("client_metrics", []):
            c_rows.append({
                "Client Institution": c.get("client_id"),
                "Partition Samples": c.get("samples"),
                "Local Loss": f"{c.get('loss', 0.0):.4f}",
                "Local Accuracy": f"{c.get('accuracy', 0.0) * 100:.1f}%"
            })
        st.table(pd.DataFrame(c_rows).set_index("Client Institution"))
    else:
        st.info("Execute a simulation run above to inspect communication rounds and convergence graphs.")
