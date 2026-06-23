import streamlit as st
import plotly.graph_objects as go

def run():
    st.title("🧠 Arsitektur Model ANN")
    st.write("---")
    
    st.markdown("""
    ### Memahami Jaringan Saraf Tiruan (Artificial Neural Network)
    
    Bagian ini menjelaskan arsitektur, keputusan desain, dan kinerja model ANN 
    yang dibangun khusus untuk memprediksi churn pelanggan.
    """)
    
    st.write("---")
    
    st.subheader("🏗️ Konfigurasi Arsitektur Jaringan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Konfigurasi Layer**
        
        | Layer | Tipe | Unit | Aktivasi | Dropout |
        |-------|------|------|----------|---------|
        | Input | Input | 6 | - | - |
        | Hidden 1 | Dense | 64 | ReLU | 0.3 |
        | Hidden 2 | Dense | 32 | ReLU | 0.2 |
        | Hidden 3 | Dense | 16 | ReLU | - |
        | Output | Dense | 1 | Sigmoid | - |
        
        **Total Parameter:** ~3,500+
        
        **Fitur Input (6):**
        - Support Calls (Panggilan Dukungan)
        - Usage Frequency (Frekuensi Penggunaan)
        - Payment Delay (Keterlambatan Pembayaran)
        - Last Interaction (Interaksi Terakhir)
        - Age (Usia)
        - Tenure (Masa Berlangganan)
        """)
    
    with col2:
        st.metric("Fitur Input", "6")
        st.metric("Hidden Layer", "3")
        st.metric("Total Layer", "5")
    
    st.markdown("""
    #### Diagram Aliran Jaringan
    
    ```
    ┌──────────────────────────────────────────────────┐
    │  INPUT LAYER (6 fitur)                           │
    │  - Support Calls                                 │
    │  - Usage Frequency                               │
    │  - Payment Delay                                 │
    │  - Last Interaction                              │
    │  - Age                                           │
    │  - Tenure                                        │
    └──────────────┬───────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────┐
    │  HIDDEN LAYER 1: 64 neuron                       │
    │  Aktivasi: ReLU (Rectified Linear Unit)          │
    │  Dropout: 0.3 (regulasi 30%)                     │
    │  Tujuan: Pelajari pola kompleks                  │
    └──────────────┬───────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────┐
    │  HIDDEN LAYER 2: 32 neuron                       │
    │  Aktivasi: ReLU                                  │
    │  Dropout: 0.2 (regulasi 20%)                     │
    │  Tujuan: Refinkan representasi yang dipelajari   │
    └──────────────┬───────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────┐
    │  HIDDEN LAYER 3: 16 neuron                       │
    │  Aktivasi: ReLU                                  │
    │  Tujuan: Transformasi fitur akhir                │
    └──────────────┬───────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────┐
    │  OUTPUT LAYER: 1 neuron                          │
    │  Aktivasi: Sigmoid (probabilitas 0-1)            │
    │  Output: Probabilitas Churn Pelanggan            │
    └──────────────────────────────────────────────────┘
    ```
    """)
    
    st.write("---")
    
    st.subheader("⚡ Fungsi Aktivasi Dijelaskan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ReLU (Rectified Linear Unit)**
        
        Digunakan di hidden layer.
        
        ```
        f(x) = max(0, x)
        ```
        
        **Mengapa ReLU?**
        - Memperkenalkan non-linearity
        - Efisien secara komputasi
        - Menghindari vanishing gradient
        - Memungkinkan jaringan mempelajari pola kompleks
        """)
    
    with col2:
        st.markdown("""
        **Sigmoid**
        
        Digunakan di output layer.
        
        ```
        f(x) = 1 / (1 + e^-x)
        ```
        
        **Mengapa Sigmoid?**
        - Output sebagai probabilitas (0-1)
        - Sempurna untuk klasifikasi biner
        - Menginterpretasi sebagai "probabilitas churn"
        """)
    
    st.write("---")
    
    st.subheader("🎓 Konfigurasi Pelatihan ⬆️ BARU & DITINGKATKAN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Optimasi & Fungsi Loss**
        
        - **Optimizer:** Adam
          - Adaptive learning rate
          - Konvergensi efisien
        
        - **Fungsi Loss:** Binary Crossentropy
          - Standar untuk klasifikasi biner
          - Mengukur error prediksi
        
        - **Metrics:**
          - Accuracy (Akurasi)
          - Precision (Presisi)
          - Recall (Recall)
        """)
    
    with col2:
        st.markdown("""
        **Regulasi & Callbacks ⬆️ BARU**
        
        - **Dropout:** 0.3, 0.2
          - Cegah overfitting
          - Kurangi co-adaptation
        
        - **Early Stopping:** ⬆️
          - Monitor validation loss
          - Patience: 7 epochs
        
        - **Learning Rate Scheduler:** ⬆️
          - Kurangi saat plateau
          - Factor: 0.5
        """)
    
    st.write("---")
    
    st.subheader("🎁 Teknik Pelatihan Baru ⬆️ DITINGKATKAN")
    
    new_techniques = st.expander("🚀 Tunjukkan Teknik Pelatihan Profesional Baru", expanded=True)
    with new_techniques:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Early Stopping** 🛑
            - Monitor: validation loss
            - Patience: 7 epochs
            - Restore: best weights
            - Manfaat: Cegah overfitting
            
            **2. Learning Rate Scheduling** 📉
            - Trigger: val_loss plateau
            - Factor: 0.5 reduction
            - Patience: 3 epochs
            - Manfaat: Optimasi lebih baik
            
            **3. Weight Initialization** 🎯
            - Hidden: HeNormal
            - Output: GlorotNormal
            - Manfaat: Konvergensi lebih cepat
            """)
        
        with col2:
            st.markdown("""
            **4. Validation Monitoring** 📊
            - Proper validation split
            - Monitor real-time
            - Early warning system
            
            **5. Class Weight Balancing** ⚖️
            - Tangani imbalans ~27% churn
            - Equal importance both classes
            - Better minority detection
            
            **Hasil:** Akurasi 86-88% dengan prediksi lebih stabil! ✨
            """)
    
    st.write("---")
    
    st.subheader("📊 Kinerja Model")
    
    tab1, tab2, tab3 = st.tabs(["Hasil Training", "Hasil Test", "Confusion Matrix"])
    
    with tab1:
        st.markdown("""
        **Classification Report - Training Set**
        
        | Metrik | Class 0 | Class 1 | Macro Avg | Weighted Avg |
        | :--- | :---: | :---: | :---: | :---: |
        | **Precision** | 0.8560 | 0.9998 | 0.9279 | 0.9376 |
        | **Recall** | 0.9997 | 0.8719 | 0.9358 | 0.9272 |
        | **F1-Score** | 0.9223 | 0.9314 | 0.9269 | 0.9275 |
        | **Support** | 136,068 | 178,566 | 314,634 | 314,634 |
        
        **Akurasi:** 92.72% ⬆️ DITINGKATKAN ✅
        
        **Insight:** 
        Kinerja seimbang di kedua class → Model belajar dengan baik!
        """)

    
    with tab2: 
        st.markdown(""" 
        **Classification Report - Test Set**

        | Metrik | Class 0 | Class 1 | Macro Avg | Weighted Avg |
        | :--- | :---: | :---: | :---: | :---: |
        | **Precision** | 0.8507 | 0.9997 | 0.9252 | 0.9351 |
        | **Recall** | 0.9997 | 0.8656 | 0.9326 | 0.9238 |
        | **F1-Score** | 0.9192 | 0.9278 | 0.9235 | 0.9241 |
        | **Support** | 28,340 | 36,983 | 65,323 | 65,323 |

        **Akurasi:** 92.38% ⬆️ DITINGKATKAN ✅

        **Insight:** 
        Kinerja train dan test sangat mirip → Tidak overfitting! 
        """)

    
    with tab3:
        st.markdown("""
        **Confusion Matrix - Test Set**
        
        ```
        Aktual / Prediksi      |    Tidak Churn |    Churn
        ───────────────────────────────────────────────────
        Tidak Churn (0)        |     28,331     |      9
        Churn (1)              |      4,971     | 32,012
        ───────────────────────────────────────────────────
        ```
        
        **Interpretasi:**
        - **True Negatives (TN):** 28,331 - Benar memprediksi tidak churn
        - **False Positives (FP):** 9 - Salah memprediksi churn (false alarm)
        - **False Negatives (FN):** 4,971 - Kasus churn yang terlewat (gagal deteksi)
        - **True Positives (TP):** 32,012 - Benar memprediksi churn
        
        **Angka-angka Penting:**
        - **Recall (Sensitivity):** 32,012 / 36,983 = **86.56%** (Kemampuan menangkap churn)
        - **Specificity:** 28,331 / 28,340 = **99.97%** (Kemampuan menghindari false alarm)
        """)

    st.write("---")
    
    st.subheader("🎯 Keputusan Desain & Alasan")
    
    st.markdown("""
    **1. Mengapa 3 Hidden Layer?**
    - Pola kompleks dalam perilaku pelanggan memerlukan multiple layers
    - Progressive feature refinement (64 → 32 → 16 neuron)
    - Kurangi dimensionality secara graceful menuju output
    
    **2. Mengapa Dropout?**
    - Cegah overfitting dengan menonaktifkan neuron secara acak
    - Paksa jaringan untuk belajar redundant representations
    - Tingkatkan generalization ke data yang belum pernah dilihat
    
    **3. Mengapa Class Weights?**
    - Dataset imbalanced (73% tidak churn vs 27% churn)
    - Balanced weights cegah bias terhadap majority class
    - Pastikan model belajar kedua class dengan baik
    
    **4. Mengapa Fitur Input Hanya 6?**
    - Menghindari data leakage
    - Gunakan hanya true behavioral predictors
    - Fitur dipilih karena terjadi SEBELUM keputusan churn
    
    **5. Mengapa Metrics Ini?**
    - **Accuracy:** Keakuratan keseluruhan
    - **Precision:** Saat model prediksi churn, benar?
    - **Recall:** Berapa % churn aktual yang tertangkap?
    - **F1-Score:** Harmonic mean (pandangan seimbang)
    """)
    
    st.write("---")
    
    st.subheader("📈 Perbandingan: Sebelum vs Sesudah Peningkatan")
    
    comparison = st.columns(2)
    
    with comparison[0]:
        st.markdown("""
        **SEBELUM (Old Training)**
        - Input: 37 fitur (berisiko leakage)
        - Akurasi: ~86%
        - Early Stopping: Tidak
        - Learning Rate: Fixed
        - Weight Init: Default
        """)
    
    with comparison[1]:
        st.markdown("""
        **SESUDAH (New Training)** ⬆️
        - Input: 6 fitur (anti-leakage)
        - Akurasi: 86-88% ⬆️
        - Early Stopping: Ya ⬆️
        - Learning Rate: Scheduled ⬆️
        - Weight Init: Professional ⬆️
        """)
    
    st.write("---")
    
    st.success("""
    **💡 Takeaway Utama:** 
    Arsitektur ANN ini menyeimbangkan kompleksitas (cukup layer untuk pola kompleks) dengan 
    kesederhanaan (tidak terlalu dalam = training cepat, interpretable). Akurasi 86-88% menunjukkan 
    bahwa model berhasil mempelajari pola churn pelanggan dengan baik.
    
    **🚀 Dengan teknik pelatihan baru, model ini siap untuk production deployment!**
    """)
    
    st.write("---")
    
    st.subheader("🔍 Pemahaman Layer-by-Layer")
    
    with st.expander("Input Layer (6 Fitur)"):
        st.markdown("""
        **Fitur yang Digunakan:**
        1. **Support Calls** - Jumlah panggilan ke layanan support
           - Indikator: Pelanggan bermasalah = risiko lebih tinggi
        
        2. **Usage Frequency** - Berapa sering pelanggan menggunakan layanan
           - Indikator: Engagement rendah = risiko lebih tinggi
        
        3. **Payment Delay** - Rata-rata keterlambatan pembayaran (hari)
           - Indikator: Masalah keuangan = risiko lebih tinggi
        
        4. **Last Interaction** - Hari terakhir ada interaksi
           - Indikator: Kontak lama = risiko lebih tinggi
        
        5. **Age** - Usia pelanggan
           - Indikator: Faktor demografis
        
        6. **Tenure** - Berapa lama menjadi pelanggan (bulan)
           - Indikator: Loyalty indicator, pelanggan lama = risiko lebih rendah
        """)
    
    with st.expander("Hidden Layers (64 → 32 → 16)"):
        st.markdown("""
        **Layer 1 (64 neuron):**
        - Menerima 6 input features
        - Belajar pola dasar dan kombinasi features
        - ReLU activation = non-linear relationships
        - Dropout 30% = mencegah co-adaptation
        
        **Layer 2 (32 neuron):**
        - Refinkan pola dari Layer 1
        - Belajar abstractions yang lebih tinggi
        - ReLU activation = lebih banyak non-linearity
        - Dropout 20% = fine-tune regularization
        
        **Layer 3 (16 neuron):**
        - Kompresi features menuju output final
        - Belajar high-level decision boundaries
        - ReLU activation = tetap dapat non-linear
        - No dropout = fokus pada prediction
        """)
    
    with st.expander("Output Layer (1 neuron)"):
        st.markdown("""
        **Sigmoid Output:**
        - Input: 16 neurons dari Hidden Layer 3
        - Activation: Sigmoid → output range [0, 1]
        - Interpretasi: Probabilitas pelanggan akan churn
        
        **Output Values:**
        - 0.0 - 0.3: Low churn risk (risiko rendah)
        - 0.3 - 0.7: Medium churn risk (risiko sedang)
        - 0.7 - 1.0: High churn risk (risiko tinggi)
        """)
    
    st.info("""
    **Struktur ini dirancang untuk:**
    ✅ Menangkap pola kompleks dalam data churn
    ✅ Generalisasi baik ke data baru
    ✅ Memberikan prediksi yang dapat dipercaya
    ✅ Mudah di-deploy dan di-maintain
    """)