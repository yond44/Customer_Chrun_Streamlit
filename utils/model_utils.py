import joblib
import json
import tensorflow as tf
import streamlit as st


@st.cache_resource
def load_random_forest_artifacts():
    try:
        rf_model = joblib.load('models/Customer_churn_rf_model.pkl')
        encoder_ordinal = joblib.load('models/encoder_ordinal.pkl')
        encoder_ohe = joblib.load('models/encoder_ohe.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/model_metadata.json', 'r') as f:
            rf_meta = json.load(f)
        return rf_model, encoder_ordinal, encoder_ohe, scaler, rf_meta
    except FileNotFoundError as e:
        st.error(f"❌ File Random Forest tidak ditemukan: {str(e)}")
        return None, None, None, None


@st.cache_resource
def load_ann_artifacts():
    try:
        model = tf.keras.models.load_model('models/ann_model.keras')
        pipeline = joblib.load('models/final_pipeline.pkl')
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        return model, pipeline, config
    except FileNotFoundError as e:
        st.error(f"❌ File Neural Network tidak ditemukan: {str(e)}")
        return None, None, None