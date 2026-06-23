import joblib
import json
import tensorflow as tf
import streamlit as st
import os
import sys


def log_status(message, status="INFO"):
    """Simple logging function"""
    timestamp = "DEBUG" if status == "DEBUG" else status
    print(f"[{timestamp}] {message}", file=sys.stderr)


@st.cache_resource
def load_random_forest_artifacts():
    log_status("Loading Random Forest artifacts...")
    
    try:
        # Load model
        log_status("  → Loading Customer_churn_rf_model.pkl")
        rf_model = joblib.load('models/Customer_churn_rf_model.pkl')
        log_status("     ✅ RF Model loaded", "SUCCESS")
        
        # Load encoders
        log_status("  → Loading encoder_ordinal.pkl")
        encoder_ordinal = joblib.load('models/encoder_ordinal.pkl')
        log_status("     ✅ Encoder ordinal loaded", "SUCCESS")
        
        log_status("  → Loading encoder_ohe.pkl")
        encoder_ohe = joblib.load('models/encoder_ohe.pkl')
        log_status("     ✅ Encoder OHE loaded", "SUCCESS")
        
        # Load scaler
        log_status("  → Loading scaler.pkl")
        scaler = joblib.load('models/scaler.pkl')
        log_status("     ✅ Scaler loaded", "SUCCESS")
        
        # Load metadata
        log_status("  → Loading model_metadata.json")
        metadata_path = 'models/model_metadata.json'
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    rf_meta = json.load(f)
                log_status(f"     ✅ Metadata loaded with keys: {list(rf_meta.keys())}", "SUCCESS")
            except json.JSONDecodeError as e:
                log_status(f"     ⚠️ JSON decode error: {e}", "WARNING")
                rf_meta = _get_default_rf_metadata()
        else:
            log_status(f"     ⚠️ model_metadata.json not found, using defaults", "WARNING")
            rf_meta = _get_default_rf_metadata()
        
        log_status("✅ Random Forest artifacts loaded successfully", "SUCCESS")
        return rf_model, encoder_ordinal, encoder_ohe, scaler, rf_meta
    
    except FileNotFoundError as e:
        log_status(f"❌ FileNotFoundError: {str(e)}", "ERROR")
        st.error(f"❌ File Random Forest tidak ditemukan:\n```\n{str(e)}\n```")
        return None, None, None, None, {}
    
    except Exception as e:
        log_status(f"❌ Unexpected error: {str(e)}", "ERROR")
        st.error(f"❌ Error loading Random Forest: {str(e)}")
        return None, None, None, None, {}


@st.cache_resource
def load_ann_artifacts():
    """
    Load ANN model dan preprocessing artifacts
    Returns: (model, pipeline, config_dict)
    """
    log_status("Loading ANN artifacts...")
    
    try:
        model_path = None
        
        if os.path.exists('models/ann_model.keras'):
            model_path = 'models/ann_model.keras'
            log_status("  → Loading ann_model.keras")
        elif os.path.exists('models/ann_model.h5'):
            model_path = 'models/ann_model.h5'
            log_status("  → Loading ann_model.h5")
        else:
            raise FileNotFoundError(
                "ANN model not found. Expected 'models/ann_model.keras' or 'models/ann_model.h5'"
            )
        
        model = tf.keras.models.load_model(model_path)
        log_status(f"     ✅ Model loaded from {model_path}", "SUCCESS")
        
        log_status("  → Loading final_pipeline.pkl")
        pipeline = joblib.load('models/final_pipeline.pkl')
        log_status("     ✅ Pipeline loaded", "SUCCESS")
        
        # Load config
        log_status("  → Loading model_config.json")
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        log_status(f"     ✅ Config loaded with keys: {list(config.keys())}", "SUCCESS")
        
        log_status("✅ ANN artifacts loaded successfully", "SUCCESS")
        return model, pipeline, config
    
    except FileNotFoundError as e:
        log_status(f"❌ FileNotFoundError: {str(e)}", "ERROR")
        st.error(f"❌ File Neural Network tidak ditemukan:\n```\n{str(e)}\n```")
        return None, None, None
    
    except Exception as e:
        log_status(f"❌ Unexpected error: {str(e)}", "ERROR")
        st.error(f"❌ Error loading ANN: {str(e)}")
        return None, None, None


def _get_default_rf_metadata():
    """Return default metadata structure"""
    return {
        'test_accuracy': None,
        'test_precision': None,
        'test_recall': None,
        'test_f1': None,
        'cv_f1_mean': None,
        'features': [],
        'feature_importance': {},
        'leakage_check': {}
    }