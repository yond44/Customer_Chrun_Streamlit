import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import tensorflow as tf
import numpy as np
from utils.model_utils import load_ann_artifacts, load_random_forest_artifacts

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    try:
        ann_model    = tf.keras.models.load_model('models/ann_model.h5')
        ann_pipeline = joblib.load('models/final_pipeline.pkl')
        with open('models/model_metadata.json', 'r') as f:
            rf_meta = json.load(f)
        with open('models/model_config.json', 'r') as f:
            ann_config = json.load(f)
        return ann_model, ann_pipeline, rf_meta, ann_config
    except FileNotFoundError as e:
        st.warning(f"⚠️ File model tidak ditemukan: {e}")
        return None, None, {}, {}


# ============================================================================
# MAIN
# ============================================================================

def run():
    ann_model, ann_pipeline, ann_config = load_ann_artifacts()
    rf_model, encoder_ordinal, encoder_ohe, scaler, rf_meta= load_random_forest_artifacts()
    

    rf_accuracy = rf_meta.get('test_accuracy', None)
    rf_f1       = rf_meta.get('test_f1',       None)
    rf_cv_f1    = rf_meta.get('cv_f1_mean',    None)
    rf_features = rf_meta.get('features',      [])
    rf_feat_imp = rf_meta.get('feature_importance', {})
    rf_leakage  = rf_meta.get('leakage_check', {})
    ann_cols    = ann_config.get('num_columns', [
        'Support Calls', 'Usage Frequency', 'Payment Delay',
        'Last Interaction', 'Age', 'Tenure'
    ])

    # ANN architecture info
    # Exclude Input layer and Dropout — only keep hidden Dense + output Dense
    if ann_model:
        all_dense = [l for l in ann_model.layers
                     if l.__class__.__name__ == 'Dense']
        ann_layers = all_dense[:-1]   # hidden layers (all Dense except last)
        ann_output = all_dense[-1]    # output layer
        ann_params = ann_model.count_params()
    else:
        ann_layers = []
        ann_output = None
        ann_params = 0

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.title("🎯 Customer Churn Prediction")
    st.markdown("#### Mini Project 2 — Artificial Neural Network | oleh Yonda Eko Dirman")
    st.write("---")

    st.markdown("""
    Aplikasi ini membandingkan dua pendekatan prediksi churn pelanggan:
    **Random Forest** (Mini Project 1) dengan banyak fitur namun berisiko *data leakage*,
    versus **Neural Network** (Mini Project 2) yang dilatih hanya dengan fitur behavioral
    berkorelasi tinggi untuk prediksi yang lebih andal.
    """)

    st.write("---")

    # ── MODEL METRICS CARDS ───────────────────────────────────────────────────
    st.subheader("📊 Performa Model")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "🌲 RF Akurasi",
            f"{rf_accuracy*100:.2f}%" if rf_accuracy else "N/A",
            "⚠️ Diduga data leakage",
            delta_color="off"
        )
    with col2:
        st.metric(
            "🌲 RF F1-Score",
            f"{rf_f1*100:.2f}%" if rf_f1 else "N/A",
            "Dari model_metadata.json",
            delta_color="off"
        )
    with col3:
        st.metric(
            "🧠 ANN Akurasi",
            "~86%",
            "✅ Lebih realistis",
            delta_color="normal"
        )
    with col4:
        st.metric(
            "🧠 ANN Parameter",
            f"{ann_params:,}" if ann_params else "N/A",
            f"{len(ann_layers)} hidden layers + 1 output",
            delta_color="off"
        )

    # Leakage alert from real metadata
    if rf_leakage.get('critical_issues'):
        st.error(
            f"⚠️ **Leakage Terdeteksi pada Random Forest** — "
            + "; ".join(rf_leakage['critical_issues'])
        )
    elif rf_accuracy and rf_accuracy > 0.99:
        st.warning(
            "⚠️ Random Forest mencapai akurasi >99% — ini mengindikasikan kemungkinan "
            "data leakage. Beberapa fitur mungkin merupakan akibat dari churn, bukan penyebabnya."
        )

    st.write("---")

    # ── COMPARISON CHART ──────────────────────────────────────────────────────
    st.subheader("📈 Perbandingan Akurasi Model")

    col1, col2 = st.columns([3, 2])

    with col1:
        rf_acc_val  = rf_accuracy  if rf_accuracy  else 0.9998
        rf_f1_val   = rf_f1        if rf_f1        else 0.9998
        ann_acc_val = 0.86
        ann_f1_val  = 0.86

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='🌲 Random Forest',
            x=['Akurasi', 'F1-Score'],
            y=[rf_acc_val, rf_f1_val],
            marker_color='#E07B54',
            text=[f"{rf_acc_val*100:.2f}%", f"{rf_f1_val*100:.2f}%"],
            textposition='outside',
            width=0.3
        ))
        fig.add_trace(go.Bar(
            name='🧠 Neural Network',
            x=['Akurasi', 'F1-Score'],
            y=[ann_acc_val, ann_f1_val],
            marker_color='#4C9BE8',
            text=[f"~{ann_acc_val*100:.0f}%", f"~{ann_f1_val*100:.0f}%"],
            textposition='outside',
            width=0.3
        ))
        fig.add_hline(
            y=0.99,
            line_dash='dash',
            line_color='red',
            annotation_text='⚠️ Ambang Batas Leakage (99%)',
            annotation_position='top left'
        )
        fig.update_layout(
            barmode='group',
            height=380,
            yaxis=dict(title='Skor', range=[0.75, 1.08]),
            xaxis_title='Metrik',
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
            margin=dict(t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Interpretasi")
        st.warning("""
            **🌲 Random Forest**
            Akurasi sangat tinggi (>99%) —
            hampir pasti karena data leakage.
            Tidak dapat diandalkan untuk
            prediksi pelanggan baru.
                    """)
        st.success("""
            **🧠 Neural Network**
            Akurasi ~86% — realistis untuk
            masalah churn nyata. Dilatih hanya
            dengan fitur behavioral langsung,
            tanpa risiko leakage.
                    """)

    st.write("---")

    # ── FEATURE COMPARISON CHART ──────────────────────────────────────────────
    st.subheader("🗂️ Fitur yang Digunakan")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance chart from real metadata
        if rf_feat_imp:
            fi_df = pd.DataFrame(
                list(rf_feat_imp.items()),
                columns=['Fitur', 'Importance']
            ).sort_values('Importance', ascending=True)

            leaky = ['Total Spend', 'Subscription Type', 'Contract Length', 'Gender']
            colors = ['#E07B54' if f in leaky else '#4C9BE8' for f in fi_df['Fitur']]

            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Fitur'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.3f}" for v in fi_df['Importance']],
                textposition='outside'
            ))
            fig_fi.update_layout(
                title="🌲 RF — Feature Importance",
                height=max(280, len(fi_df) * 36),
                xaxis_title='Importance',
                margin=dict(l=5, r=60, t=40, b=30)
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("🔵 Biru = fitur aman | 🟠 Oranye = potensi leakage")
        else:
            st.info("Feature importance akan tampil setelah model_metadata.json dimuat.")

    with col2:
        # ANN features as a clean donut
        ann_feat_labels = ann_cols if ann_cols else [
            'Support Calls', 'Usage Frequency', 'Payment Delay',
            'Last Interaction', 'Age', 'Tenure'
        ]
        fig_ann = go.Figure(go.Bar(
            x=[1] * len(ann_feat_labels),
            y=ann_feat_labels,
            orientation='h',
            marker_color='#4C9BE8',
            text=['✅ Korelasi Tinggi'] * len(ann_feat_labels),
            textposition='inside',
            textfont=dict(color='white', size=11)
        ))
        fig_ann.update_layout(
            title="🧠 ANN — Fitur Terpilih",
            height=max(280, len(ann_feat_labels) * 48),
            xaxis=dict(visible=False),
            margin=dict(l=5, r=10, t=40, b=30)
        )
        st.plotly_chart(fig_ann, use_container_width=True)
        st.caption(f"ANN hanya menggunakan {len(ann_feat_labels)} fitur dengan korelasi behavioral langsung ke churn.")

    st.write("---")

    # ── ANN ARCHITECTURE VISUAL ───────────────────────────────────────────────
    st.subheader("🧠 Arsitektur Neural Network")

    if ann_layers:
        layer_data = []
        layer_data.append({
            'Layer': 'Input',
            'Neurons': len(ann_cols),
            'Aktivasi': '—',
            'Keterangan': f'{len(ann_cols)} fitur input (scaled)'
        })
        for i, l in enumerate(ann_layers):
            act = l.get_config().get('activation', 'relu').capitalize()
            layer_data.append({
                'Layer': f'Hidden {i+1}',
                'Neurons': l.units,
                'Aktivasi': act,
                'Keterangan': f'Dense({l.units}, {act}) + Dropout'
            })
        out_act = ann_output.get_config().get('activation', 'sigmoid').capitalize() if ann_output else 'Sigmoid'
        layer_data.append({
            'Layer': 'Output',
            'Neurons': 1,
            'Aktivasi': out_act,
            'Keterangan': 'Probabilitas churn (0–1)'
        })

        arch_df = pd.DataFrame(layer_data)

        # Lollipop chart for neurons per layer
        fig_arch = go.Figure()
        fig_arch.add_trace(go.Scatter(
            x=arch_df['Layer'],
            y=arch_df['Neurons'],
            mode='lines+markers',
            marker=dict(size=arch_df['Neurons'].apply(lambda x: min(x/2, 40) + 10), color='#4C9BE8', line=dict(width=2, color='white')),
            line=dict(color='#4C9BE8', width=2),
            text=[f"{r['Layer']}<br>{r['Neurons']} neuron<br>{r['Aktivasi']}" for _, r in arch_df.iterrows()],
            hoverinfo='text'
        ))
        fig_arch.update_layout(
            height=300,
            yaxis_title='Jumlah Neuron',
            xaxis_title='Layer',
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_arch, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(arch_df, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Parameter", f"{ann_params:,}")
            st.metric("Hidden Layers", len(ann_layers))
            st.metric("Optimizer", "Adam")
            st.metric("Loss Function", "Binary Crossentropy")

    st.write("---")

    # ── NAVIGATION GUIDE ──────────────────────────────────────────────────────
    st.subheader("🗺️ Panduan Navigasi")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**📈 Analisis Data**\n\nEksplorasi dataset, preprocessing, distribusi fitur, dan penanganan outlier.")
    with col2:
        st.info("**🧠 Arsitektur Model**\n\nDetail layer ANN, fungsi aktivasi, riwayat training, dan performa model.")
    with col3:
        st.info("**⚖️ Perbandingan Model**\n\nANN vs Random Forest — metrik, leakage risk, feature importance.")
    with col4:
        st.info("**🎯 Prediksi**\n\nMasukkan data pelanggan dan bandingkan hasil prediksi kedua model.")

    st.write("---")
    st.caption("Mini Project 2 — ANN Model | Yonda Eko Dirman")