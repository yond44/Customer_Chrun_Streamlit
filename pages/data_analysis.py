import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import tensorflow as tf
from utils.model_utils import load_ann_artifacts, load_random_forest_artifacts

# ============================================================================
# LOAD ARTIFACTS FOR REAL DATA
# ============================================================================

@st.cache_resource
def load_artifacts():
    try:
        rf_model      = joblib.load('models/Customer_churn_rf_model.pkl')
        ann_pipeline  = joblib.load('models/final_pipeline.pkl')
        with open('models/model_metadata.json', 'r') as f:
            rf_meta = json.load(f)
        with open('models/model_config.json', 'r') as f:
            ann_config = json.load(f)
        return rf_model, ann_pipeline, rf_meta, ann_config
    except FileNotFoundError as e:
        st.warning(f"⚠️ Beberapa file model tidak ditemukan: {e}")
        return None, None, {}, {}


# ============================================================================
# MAIN
# ============================================================================

def run():
    st.title("📈 Analisis Data")
    st.write("---")

    st.markdown("""
    ### Gambaran Umum Dataset
    Halaman ini menjelaskan dataset churn pelanggan yang digunakan untuk melatih kedua model,
    termasuk proses pembersihan data, encoding, dan scaling yang diterapkan.
    """)

    rf_model, ann_pipeline, rf_meta, ann_config = load_artifacts()

    # Pull real values from metadata where available
    rf_features     = rf_meta.get('features', [])
    rf_feat_imp     = rf_meta.get('feature_importance', {})
    ann_num_cols    = ann_config.get('num_columns', [])

    # ========== METRIC CARDS ==========

    st.subheader("📊 Statistik Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Mentah", "~440.000", "Baris sebelum pembersihan")
    with col2:
        st.metric("Total Data Bersih", "~435.000", "Setelah hapus duplikat & outlier")
    with col3:
        n_feat = len(rf_features) if rf_features else 10
        st.metric("Jumlah Fitur", str(n_feat), "Setelah encoding (RF)")
    with col4:
        st.metric("Kelas Target", "2", "Churn / Tidak Churn")

    st.write("---")

    # ========== FITUR YANG DIGUNAKAN ==========
    st.subheader("🗂️ Fitur yang Digunakan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔵 Fitur Bersama — Digunakan Kedua Model")
        shared_features = {
            'Fitur': ann_num_cols if ann_num_cols else [
                'Support Calls', 'Usage Frequency', 'Payment Delay',
                'Last Interaction', 'Age', 'Tenure'
            ],
            'Tipe': ['Numerik'] * (len(ann_num_cols) if ann_num_cols else 6),
            'Keterangan': [
                'Jumlah panggilan ke layanan pelanggan (3 bulan)',
                'Frekuensi penggunaan layanan per bulan',
                'Rata-rata keterlambatan pembayaran (hari)',
                'Hari sejak interaksi terakhir dengan layanan',
                'Usia pelanggan',
                'Lama berlangganan (bulan)'
            ][:len(ann_num_cols) if ann_num_cols else 6]
        }
        st.dataframe(pd.DataFrame(shared_features), hide_index=True, use_container_width=True)
        st.success("✅ Fitur-fitur ini memiliki korelasi langsung dan tinggi terhadap churn.")

    with col2:
        st.markdown("##### 🌲 Fitur Eksklusif Random Forest")
        rf_only = {
            'Fitur': ['Gender', 'Subscription Type', 'Contract Length', 'Total Spend'],
            'Tipe': ['Kategorik (OHE)', 'Kategorik (Ordinal)', 'Kategorik (Ordinal)', 'Numerik'],
            'Keterangan': [
                'Jenis kelamin pelanggan',
                'Jenis paket langganan (Basic/Standard/Premium)',
                'Durasi kontrak (Monthly/Quarterly/Annual)',
                'Total pengeluaran pelanggan ($)'
            ]
        }
        st.dataframe(pd.DataFrame(rf_only), hide_index=True, use_container_width=True)
        st.warning("⚠️ Beberapa fitur ini diduga mengandung data leakage — bisa jadi *akibat* dari churn, bukan penyebabnya.")

    st.write("---")

    # ========== PREPROCESSING STEPS ==========
    st.subheader("🔧 Langkah-Langkah Preprocessing")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Nilai Hilang", "Outlier", "Encoding", "Scaling"
    ])

    with tab1:
        st.markdown("""
        #### Penanganan Nilai Hilang (Missing Values)

        **Kolom yang memiliki nilai hilang:**
        - `Contract Length`
        - `Subscription Type`
        - `Gender`

        **Metode yang digunakan:**
        - Nilai hilang diisi dengan **modus (nilai terbanyak)** dari setiap kolom
        - Dilakukan sebelum proses split data agar tidak terjadi data leakage
        - Setelah imputasi: **0 nilai hilang** di seluruh kolom

        ```python
        for col in columns_with_missing:
            df[col] = df[col].fillna(df[col].mode()[0])
        ```
        """)

    with tab2:
        st.markdown("""
        #### Penghapusan Outlier (Metode IQR / Tukey)

        Outlier dihapus menggunakan metode **Interquartile Range (IQR)**
        pada semua kolom numerik:

        ```
        Q1  = persentil ke-25
        Q3  = persentil ke-75
        IQR = Q3 - Q1

        Batas Bawah = Q1 - 1.5 × IQR
        Batas Atas  = Q3 + 1.5 × IQR

        Data di luar batas → dihapus
        ```

        **Kolom numerik yang dicek:**
        - Age, Tenure, Usage Frequency
        - Support Calls, Payment Delay
        - Total Spend, Last Interaction

        Setelah penghapusan outlier: **~435.000 baris** tersisa dari ~440.000 data awal.
        """)

    with tab3:
        st.markdown("""
        #### Strategi Encoding Fitur Kategorik

        **Ordinal Encoding** — digunakan untuk fitur dengan urutan yang bermakna:
        """)

        ordinal_df = pd.DataFrame({
            'Fitur': ['Contract Length', 'Subscription Type'],
            'Urutan Encoding': [
                'Monthly (0) → Quarterly (1) → Annual (2)',
                'Basic (0) → Standard (1) → Premium (2)'
            ],
            'Alasan': [
                'Kontrak lebih panjang = komitmen lebih tinggi',
                'Paket lebih tinggi = nilai lebih tinggi'
            ]
        })
        st.dataframe(ordinal_df, hide_index=True, use_container_width=True)

        st.markdown("""
        **One-Hot Encoding (OHE)** — digunakan untuk fitur tanpa urutan:
        """)

        ohe_df = pd.DataFrame({
            'Fitur Asli': ['Gender'],
            'Hasil Encoding': ['Gender_Female, Gender_Male'],
            'Alasan': ['Tidak ada urutan antara Male dan Female']
        })
        st.dataframe(ohe_df, hide_index=True, use_container_width=True)

    with tab4:
        st.markdown("""
        #### Scaling Fitur Numerik

        **StandardScaler** digunakan untuk semua fitur numerik:

        ```
        nilai_scaled = (nilai - rata_rata) / standar_deviasi
        ```

        **Hasil:** Setiap fitur memiliki **rata-rata = 0** dan **standar deviasi = 1**

        **Mengapa penting?**
        - Mencegah fitur dengan nilai besar (misal: Total Spend dalam ratusan)
          mendominasi fitur kecil (misal: Support Calls 0–15)
        - Wajib untuk Neural Network agar gradient descent bekerja optimal
        - Juga diterapkan pada Random Forest untuk konsistensi pipeline

        **Implementasi:**
        - RF: `StandardScaler` diterapkan langsung pada kolom numerik
        - ANN: `StandardScaler` diintegrasikan dalam `ColumnTransformer` pipeline
        """)

    st.write("---")

    # ========== DISTRIBUSI TARGET ==========
    st.subheader("🎯 Distribusi Variabel Target (Churn)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Distribusi Kelas (Setelah Pembersihan)**

        | Kelas | Label | Jumlah (perkiraan) | Persentase |
        |-------|-------|-------------------|-----------|
        | 0 | Tidak Churn | ~358.000 | ~82% |
        | 1 | Churn | ~77.000 | ~18% |

        **Ketidakseimbangan kelas** ditangani dengan:
        - **ANN:** `compute_class_weight('balanced')` → bobot kelas dihitung otomatis
        - **RF:** `class_weight='balanced'` pada parameter model

        Pendekatan ini memastikan model tidak bias terhadap kelas mayoritas
        (pelanggan yang tidak churn).
        """)

    with col2:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Tidak Churn (0)', 'Churn (1)'],
            values=[82, 18],
            hole=0.4,
            marker=dict(colors=['#4C9BE8', '#E8744C']),
            textinfo='label+percent',
            textfont_size=13
        )])
        fig_pie.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.write("---")

    # ========== FEATURE IMPORTANCE ==========
    if rf_feat_imp:
        st.subheader("📊 Feature Importance — Random Forest")
        st.caption("Nilai importance diambil langsung dari model RF yang tersimpan (`model_metadata.json`).")

        fi_df = pd.DataFrame(
            list(rf_feat_imp.items()),
            columns=['Fitur', 'Importance']
        ).sort_values('Importance', ascending=True)

        # Color leakage-prone features differently
        leaky = ['Total Spend', 'Subscription Type', 'Contract Length', 'Gender']
        colors = ['#E8744C' if f in leaky else '#4C9BE8' for f in fi_df['Fitur']]

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Fitur'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.4f}" for v in fi_df['Importance']],
            textposition='outside'
        ))
        fig_fi.update_layout(
            height=max(300, len(fi_df) * 38),
            xaxis_title='Skor Importance',
            margin=dict(l=10, r=80, t=10, b=40)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔵 **Biru** = Fitur dengan korelasi tinggi ke churn (digunakan ANN & RF)")
        with col2:
            st.markdown("🟠 **Oranye** = Fitur eksklusif RF yang berpotensi data leakage")

        st.write("---")

    # ========== PIPELINE ANN ==========
    st.subheader("⚙️ Pipeline Preprocessing ANN")

    ann_cols = ann_num_cols if ann_num_cols else [
        'Support Calls', 'Usage Frequency', 'Payment Delay',
        'Last Interaction', 'Age', 'Tenure'
    ]

    st.markdown(f"""
    Pipeline ANN menggunakan `ColumnTransformer` dengan tahapan berikut:

    ```python
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),   # Isi nilai hilang dengan median
        StandardScaler()                    # Standarisasi fitur numerik
    )

    final_pipeline = ColumnTransformer([
        ('pipe_num', num_pipeline, {ann_cols})
    ])
    ```

    **Kolom yang diproses oleh pipeline:**
    {', '.join(f'`{c}`' for c in ann_cols)}

    Pipeline ini disimpan sebagai `final_pipeline.pkl` dan digunakan langsung
    saat prediksi untuk memastikan transformasi yang konsisten dengan data latih.
    """)

    st.write("---")

    # ========== TRAIN/VAL/TEST SPLIT ==========
    st.subheader("✂️ Pembagian Data (Train / Validation / Test)")

    split_df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'Proporsi': ['~72%', '~13%', '15%'],
        'Keterangan': [
            'Digunakan untuk melatih model',
            'Digunakan untuk memantau performa selama training (Early Stopping)',
            'Digunakan untuk evaluasi akhir model (test_size=0.15, random_state=42)'
        ]
    })
    st.dataframe(split_df, hide_index=True, use_container_width=True)

    st.markdown("""
    ```python
    # Pembagian data ANN (MP02)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42
    )
    ```
    """)

    st.write("---")

    st.info("""
    **Catatan Penting:** Dataset yang sama (`customer_churn_with_outliers.csv`) digunakan
    oleh kedua model. Perbedaan utama terletak pada **pemilihan fitur** — Random Forest
    menggunakan semua 10 fitur, sementara Neural Network hanya menggunakan 6 fitur
    yang memiliki korelasi behavioral langsung dengan churn, tanpa risiko data leakage.
    """)