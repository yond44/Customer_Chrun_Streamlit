import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
import json
import tensorflow as tf
from pathlib import Path
from utils.model_utils import load_ann_artifacts,load_random_forest_artifacts





def run():
    st.title("🎯 Prediksi Customer Churn")
    st.write("---")

    st.markdown("""
    Bandingkan bagaimana dua model memprediksi apakah pelanggan akan pergi atau tidak.
    
    **Model 1 (Random Forest):** Menggunakan 10 fitur termasuk beberapa yang mungkin tidak begitu relevan.
    
    **Model 2 (Neural Network):** Hanya menggunakan 6 fitur terbaik yang benar-benar penting untuk prediksi.
    """)

    rf_model, encoder_ordinal, encoder_ohe, scaler, meta = load_random_forest_artifacts()
    ann_model, ann_pipeline, ann_config = load_ann_artifacts()

    if rf_model is None or ann_model is None:
        st.stop()

    with st.form("prediction_form", border=True):
        st.subheader("📋 Informasi Pelanggan")

        st.markdown("#### 🔵 Fitur yang Digunakan Kedua Model")
        st.caption("6 fitur penting ini digunakan oleh kedua model untuk prediksi.")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Umur", min_value=18, max_value=80, value=35)
            tenure = st.number_input("Lama Berlangganan (bulan)", min_value=0, max_value=72, value=12, step=1)
        with col2:
            usage_frequency = st.number_input("Frekuensi Penggunaan (kali/bulan)", min_value=1, max_value=30, value=15, step=1)
            support_calls = st.number_input("Panggilan Dukungan (3 bulan terakhir)", min_value=0, max_value=15, value=2, step=1)
        with col3:
            payment_delay = st.number_input("Keterlambatan Pembayaran (hari)", min_value=0, max_value=60, value=10, step=1)
            last_interaction = st.number_input("Hari Sejak Interaksi Terakhir", min_value=0, max_value=60, value=10, step=1)

        st.write("---")

        st.markdown("#### 🌲 Hanya Random Forest - Fitur Tambahan")
        st.caption("4 fitur ini hanya digunakan oleh Random Forest saja.")

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with col5:
            subscription_type = st.selectbox("Tipe Paket", ["Basic", "Standard", "Premium"])
        with col6:
            contract_length = st.selectbox("Durasi Kontrak", ["Bulanan", "Triwulan", "Tahunan"])
        with col7:
            total_spend = st.number_input("Total Pengeluaran ($)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0, format="%.2f")

        st.write("---")

        col_rf, col_ann = st.columns(2)
        with col_rf:
            st.warning(
                "**🌲 Random Forest — 10 Fitur**\n\n"
                "Umur, Lama Berlangganan, Frekuensi Penggunaan, Panggilan Dukungan, "
                "Keterlambatan Pembayaran, Hari Terakhir Interaksi, Jenis Kelamin, "
                "Tipe Paket, Durasi Kontrak, Total Pengeluaran\n\n"
                "⚠️ Beberapa fitur mungkin bukan penyebab asli customer churn."
            )
        with col_ann:
            st.success(
                "**🧠 Neural Network — 6 Fitur**\n\n"
                "Panggilan Dukungan, Frekuensi Penggunaan, Keterlambatan Pembayaran, "
                "Hari Terakhir Interaksi, Umur, Lama Berlangganan\n\n"
                "✅ Hanya fitur terbaik yang benar-benar mempengaruhi churn."
            )

        submit_button = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            use_container_width=True,
            type="primary"
        )

    st.write("---")

    if submit_button:

        try:
            ordinal_encoded = encoder_ordinal.transform([[subscription_type, contract_length]])
            ordinal_df = pd.DataFrame(ordinal_encoded, columns=['Subscription Type', 'Contract Length'])

            ohe_encoded = encoder_ohe.transform([[gender]])
            ohe_df = pd.DataFrame(ohe_encoded, columns=encoder_ohe.get_feature_names_out(['Gender']))

            numeric_input = pd.DataFrame([{
                'Age': age,
                'Tenure': tenure,
                'Usage Frequency': usage_frequency,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Total Spend': total_spend,
                'Last Interaction': last_interaction
            }])
            numeric_scaled = scaler.transform(numeric_input)
            numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_input.columns)

            rf_features_final = pd.concat([ordinal_df, ohe_df, numeric_df], axis=1)
            rf_prob = float(rf_model.predict_proba(rf_features_final)[0][1])

        except Exception as e:
            st.error(f"❌ Erro prediksi Random Forest: {e}")
            st.stop()

        try:
            ann_input = pd.DataFrame([{
                'Support Calls': support_calls,
                'Usage Frequency': usage_frequency,
                'Payment Delay': payment_delay,
                'Last Interaction': last_interaction,
                'Age': age,
                'Tenure': tenure
            }])
            ann_features_scaled = ann_pipeline.transform(ann_input)
            ann_prob = float(ann_model.predict(ann_features_scaled, verbose=0)[0][0])

        except Exception as e:
            st.error(f"❌ Erro prediksi Neural Network: {e}")
            st.stop()

        rf_churn     = rf_prob  >= 0.5
        ann_churn    = ann_prob >= 0.5
        avg_prob     = (rf_prob + ann_prob) / 2
        diff         = abs(rf_prob - ann_prob)
        models_agree = rf_churn == ann_churn

        st.subheader("🎯 Hasil Prediksi")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            st.markdown("#### 🌲 Random Forest")
            if rf_churn:
                st.error("⚠️ **AKAN PERGI**")
            else:
                st.success("✅ **AKAN TETAP**")

        with col2:
            st.write("")
            st.write("")
            if models_agree:
                st.success("🤝\n\n**Sepakat**")
            else:
                st.warning("🔀\n\n**Berbeda**")

        with col3:
            st.markdown("#### 🧠 Neural Network")
            if ann_churn:
                st.error("⚠️ **AKAN PERGI**")
            else:
                st.success("✅ **AKAN TETAP**")

        st.write("---")

        st.subheader("📊 Penilaian Risiko")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Perbedaan Prediksi",
                f"{diff*100:.1f}%",
                help="Semakin kecil perbedaan, semakin dapat dipercaya prediksinya."
            )
        with col2:
            if avg_prob > 0.7:
                risk_level = "🔴 SANGAT TINGGI"
            elif avg_prob > 0.4:
                risk_level = "🟠 SEDANG"
            else:
                risk_level = "🟢 RENDAH"
            st.metric("Tingkat Risiko", risk_level)

        if models_agree:
            if avg_prob > 0.5:
                st.error("✅ Kedua model sepakat pelanggan ini **AKAN PERGI**. Lakukan aksi retensi sekarang!")
            else:
                st.success("✅ Kedua model sepakat pelanggan ini **AKAN TETAP**. Tingkat kepercayaan tinggi.")
        else:
            st.warning(
                "⚠️ Kedua model berbeda prediksi. Ini terjadi karena Random Forest menggunakan fitur tambahan "
                "yang kurang relevan.\n\n"
                "**Percayai Neural Network** — model ini hanya menggunakan fitur-fitur terbaik tanpa fitur yang "
                "tidak penting."
            )

        st.write("---")

        st.subheader("👥 Ringkasan Profil Pelanggan")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({
                'Atribut': ['Umur', 'Lama Berlangganan', 'Jenis Kelamin', 'Tipe Paket', 'Durasi Kontrak'],
                'Nilai': [
                    f"{age} tahun",
                    f"{tenure} bulan ({tenure/12:.1f} tahun)",
                    gender, subscription_type, contract_length
                ]
            }), hide_index=True, use_container_width=True)

        with col2:
            st.dataframe(pd.DataFrame({
                'Perilaku': ['Frekuensi Penggunaan', 'Panggilan Dukungan', 'Keterlambatan Pembayaran', 'Total Pengeluaran', 'Hari Terakhir Interaksi'],
                'Nilai': [
                    f"{usage_frequency} kali/bulan",
                    f"{support_calls} panggilan",
                    f"{payment_delay} hari",
                    f"${total_spend:.2f}",
                    f"{last_interaction} hari yang lalu"
                ]
            }), hide_index=True, use_container_width=True)

        st.write("---")
        st.subheader("🔍 Faktor-Faktor Utama")

        risk_factors = []

        if tenure < 6:
            risk_factors.append(("Pelanggan baru (< 6 bulan)", "Risiko tinggi", 0.7))
        elif tenure < 12:
            risk_factors.append(("Pelanggan awal (< 1 tahun)", "Risiko sedang", 0.5))
        elif tenure > 48:
            risk_factors.append(("Pelanggan setia (> 4 tahun)", "Risiko rendah", 0.2))

        if support_calls > 5:
            risk_factors.append(("Banyak panggilan dukungan (> 5)", "Ada masalah yang dihadapi", 0.6))
        if payment_delay > 20:
            risk_factors.append(("Sering terlambat bayar", "Masalah finansial", 0.5))
        if usage_frequency < 10:
            risk_factors.append(("Penggunaan rendah", "Kurang engagement", 0.4))
        if contract_length == "Bulanan":
            risk_factors.append(("Kontrak bulanan", "Mudah untuk berhenti", 0.4))
        elif contract_length == "Tahunan":
            risk_factors.append(("Kontrak tahunan", "Komitmen jangka panjang", 0.2))
        if subscription_type == "Premium":
            risk_factors.append(("Paket Premium", "Pelanggan bernilai tinggi", 0.3))
        elif subscription_type == "Basic":
            risk_factors.append(("Paket Basic", "Komitmen rendah", 0.5))

        if risk_factors:
            for factor, reason, impact in risk_factors:
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.write(f"**{factor}**")
                with col2:
                    st.caption(reason)
                with col3:
                    if impact >= 0.6:
                        st.write("🔴 Tinggi")
                    elif impact >= 0.4:
                        st.write("🟡 Sedang")
                    else:
                        st.write("🟢 Rendah")
        else:
            st.caption("Tidak ada faktor risiko yang perlu dikhawatirkan untuk pelanggan ini.")

        st.write("---")
        st.subheader("⚖️ Perbandingan Model")

        st.dataframe(pd.DataFrame({
            'Aspek': [
                'Algoritma',
                'Jumlah Data Latih',
                'Jumlah Fitur',
                'Hasil Prediksi',
                'Akurasi Latihan',
                'Risiko Kesalahan',
                'Tingkat Kepercayaan'
            ],
            'Random Forest (MP01)': [
                'Ensemble — 200 pohon',
                '~440 ribu pelanggan',
                '10 fitur (7 angka + 3 kategori)',
                '⚠️ AKAN PERGI' if rf_churn else '✅ AKAN TETAP',
                '99.99% — terlalu sempurna',
                '🔴 TINGGI — pakai fitur tidak penting',
                '🟡 SEDANG'
            ],
            'Neural Network (MP02)': [
                'Deep Learning — 3 lapisan',
                '~435 ribu pelanggan',
                '6 fitur (hanya yang penting)',
                '⚠️ AKAN PERGI' if ann_churn else '✅ AKAN TETAP',
                '~86% — realistis',
                '🟢 RENDAH — pakai fitur terbaik',
                '🟢 TINGGI'
            ]
        }), hide_index=True, use_container_width=True)

        st.write("---")
        st.subheader("🔍 Mengapa Kedua Model Bisa Berbeda Hasil?")

        col1, col2 = st.columns(2)
        with col1:
            st.warning("""
            **🌲 Random Forest — 10 Fitur**

            Dilatih menggunakan semua fitur yang ada termasuk Jenis Kelamin, Tipe Paket,
            Durasi Kontrak, dan Total Pengeluaran.

            Beberapa fitur ini **mungkin tidak relevan** — bisa jadi adalah akibat dari 
            keputusan churn, bukan penyebabnya. Ini menjelaskan mengapa akurasi latihnya 
            sempurna (~99.99%), yang jarang terjadi di dunia nyata.
            """)
        with col2:
            st.success("""
            **🧠 Neural Network — 6 Fitur**

            Dilatih hanya dengan fitur-fitur terbaik yang benar-benar berkorelasi dengan churn:
            Panggilan Dukungan, Frekuensi Penggunaan, Keterlambatan Pembayaran, Hari Terakhir 
            Interaksi, Umur, dan Lama Berlangganan.

            Tanpa fitur yang tidak penting. ANN mencapai akurasi ~86% — hasil yang realistis 
            dan dapat dipercaya untuk memprediksi pelanggan baru.
            """)

        st.write("---")
        st.subheader("🎯 Model Mana Yang Harus Dipercaya?")

        if models_agree:
            st.success(
                "✅ Kedua model setuju — prediksi Neural Network diperkuat oleh Random Forest. "
                "Anda bisa percaya hasil ini dan mengambil tindakan."
            )
        else:
            st.warning(
                f"⚠️ Kedua model berbeda ({diff*100:.1f}% gap). **Percayai Neural Network** — "
                "model ini dirancang khusus tanpa fitur-fitur yang tidak penting, sehingga prediksinya "
                "lebih akurat untuk pelanggan baru."
            )

        with st.expander("📚 Apa itu Fitur Tidak Penting dan Mengapa Penting Diketahui?"):
            st.markdown("""
            ### Apa Itu Fitur Tidak Penting?

            Fitur tidak penting adalah informasi yang adalah **akibat dari churn**, bukan **penyebab**. 
            Model belajar mengenali pola ini dan mendapat akurasi sempurna saat latihan — 
            tapi gagal saat menghadapi data pelanggan baru yang belum churn.

            ### Contoh
            ❌ Tidak Penting: Total Pengeluaran — pelanggan yang sudah memutuskan untuk pergi
            biasanya mengurangi pembelian terlebih dahulu. Ini akibat, bukan sebab.

            ✅ Penting: Panggilan Dukungan, Keterlambatan Pembayaran — mencerminkan
            perilaku nyata pelanggan sebelum keputusan churn.
            ### Mengapa Random Forest Akurasi 99.99%?

            RF menggunakan 10 fitur termasuk yang tidak penting. Model "melihat jawabannya" 
            saat latihan dan menjadi terlalu pintar dengan data itu saja.

            ### Mengapa Neural Network 86% Lebih Dipercaya?

            ANN dibatasi hanya 6 fitur terbaik yang benar-benar penting. 86% adalah akurasi 
            yang realistis — dan hasilnya jauh lebih baik saat memprediksi pelanggan baru.
            """)

        st.write("---")
        st.subheader("💡 Saran Tindakan")

        recommendations = generate_recommendations(
            avg_prob, rf_prob, ann_prob, age, tenure, usage_frequency,
            support_calls, payment_delay, contract_length, subscription_type
        )

        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}. {rec}**")


def generate_recommendations(avg_prob, rf_prob, ann_prob, age, tenure, usage_freq,
                              support_calls, payment_delay, contract, subscription):
    recommendations = []
    disagreement = abs(rf_prob - ann_prob)
    ann_churn = ann_prob >= 0.5

    if disagreement > 0.15:
        recommendations.append("Kedua model berbeda — gunakan prediksi Neural Network untuk pengambilan keputusan.")
    else:
        recommendations.append("Kedua model sepakat — tingkat kepercayaan hasil prediksi sangat tinggi.")

    if ann_churn:
        if avg_prob > 0.7:
            recommendations.append("🚨 RISIKO SANGAT TINGGI: Lakukan kampanye retensi segera untuk pelanggan ini.")
        else:
            recommendations.append("⚡ RISIKO TINGGI: Hubungi pelanggan proaktif sebelum mereka memutuskan pergi.")
        if support_calls > 5:
            recommendations.append("📞 Banyak panggilan dukungan — berikan dedicated account manager untuk mengatasi masalah.")
    else:
        recommendations.append("✨ Pelanggan ini akan tetap. Fokus pada peningkatan loyalitas dan engagement.")
        if subscription == "Basic":
            recommendations.append("🎁 Tawarkan upgrade ke Premium dengan insentif khusus.")

    if tenure < 6:
        recommendations.append("🆕 Pelanggan baru — tingkatkan onboarding dan dukungan di tahap awal.")
    elif tenure > 24:
        recommendations.append("🏆 Pelanggan setia lama — berikan apresiasi dengan VIP benefits atau penawaran eksklusif.")

    if payment_delay > 20:
        recommendations.append("💳 Ada keterlambatan pembayaran — tawarkan opsi pembayaran yang lebih fleksibel.")

    if contract == "Bulanan":
        recommendations.append("📅 Kontrak bulanan — tawarkan insentif untuk upgrade ke paket jangka panjang.")

    return recommendations[:5]