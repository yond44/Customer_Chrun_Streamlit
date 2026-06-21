import joblib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras


class ModelLoader: 
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.pipeline = None
        self.config = None
    
    def load_model(self, model_path="ann_model.h5"):
        try:
            full_path = self.model_dir / model_path
            self.model = keras.models.load_model(str(full_path))
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_pipeline(self, pipeline_path="final_pipeline.pkl"):
        try:
            full_path = self.model_dir / pipeline_path
            self.pipeline = joblib.load(str(full_path))
            return True
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return False
    
    def load_config(self, config_path="model_config.json"):
        try:
            full_path = self.model_dir / config_path
            with open(full_path, 'r') as f:
                self.config = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def load_all(self):
        success = True
        success &= self.load_model()
        success &= self.load_pipeline()
        success &= self.load_config()
        return success


class Preprocessor:
    def __init__(self, pipeline, config):
        self.pipeline = pipeline
        self.config = config
        self.num_columns = config.get('num_columns', [])
    
    def preprocess(self, data):
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data_processed = self.pipeline.transform(data)
            else:
                data_processed = self.pipeline.transform(data)
            
            return data_processed
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None


class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, data):
        try:
            # Preprocess data
            data_processed = self.preprocessor.preprocess(data)
            
            if data_processed is None:
                return {'error': 'Preprocessing failed'}
            
            # Make prediction
            probability = self.model.predict(data_processed, verbose=0)[0][0]
            
            # Determine class
            predicted_class = 1 if probability > 0.5 else 0
            
            return {
                'success': True,
                'probability': float(probability),
                'predicted_class': int(predicted_class),
                'churn': 'Yes' if predicted_class == 1 else 'No',
                'confidence': float(probability) if predicted_class == 1 else float(1 - probability)
            }
        
        except Exception as e:
            return {'error': str(e), 'success': False}


def create_feature_importance_data(model_config):
    features = model_config.get('num_columns', [])
    return features


def get_model_insights(model_config):
    insights = {
        'input_dim': model_config.get('input_dim'),
        'num_numeric_features': len(model_config.get('num_columns', [])),
        'ordinal_features': list(model_config.get('ordinal_categories', {}).keys()),
        'date_trained': model_config.get('date_trained', 'Unknown')
    }
    return insights
