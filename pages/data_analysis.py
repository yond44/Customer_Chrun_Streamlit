import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def run():
    st.title("📈 Data Analysis")
    st.write("---")
    
    st.markdown("""
    ### Dataset Overview
    
    This section explores the customer churn dataset used for model training.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", "304,752", "After Cleaning")
    with col2:
        st.metric("Features", "18", "Input Variables")
    with col3:
        st.metric("Target Classes", "2", "Churn / No Churn")
    with col4:
        st.metric("Missing Values", "0", "After Imputation")
    
    st.write("---")
    
    st.subheader("🔧 Data Preprocessing Steps")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Outliers", "Encoding", "Scaling"])
    
    with tab1:
        st.markdown("""
        **Missing Values Handling**
        
        - Identified columns with missing values
        - Used **mode imputation** for categorical features
        - Applied **median imputation** in preprocessing pipeline
        
        Example columns handled:
        - Contract Length
        - Subscription Type
        - Gender
        """)
    
    with tab2:
        st.markdown("""
        **Outlier Removal (Z-Score)**
        
        Applied to normally distributed numeric columns:
        - Used 3-sigma rule (mean ± 3*std)
        - Identified and removed extreme values
        
        Numeric Features Checked:
        - Total Charges
        - Monthly Charges
        - Customer Service Calls
        """)
    
    with tab3:
        st.markdown("""
        **Feature Encoding Strategy**
        
        **Ordinal Encoding** (preserves order):
        - Contract Length: Monthly < Quarterly < Annual
        - Subscription Type: Basic < Standard < Premium
        
        **One-Hot Encoding** (categorical):
        - Gender: Male / Female
        """)
    
    with tab4:
        st.markdown("""
        **Feature Scaling**
        
        - **StandardScaler** for numeric features
        - Transforms data to mean=0, std=1
        - Essential for neural network performance
        - Prevents large values from dominating learning
        """)
    
    st.write("---")
    
    st.subheader("📋 Feature Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Numeric Features (Scaled)**
        - Customer Age
        - Monthly Charges
        - Total Charges
        - Customer Service Calls
        - Tenure
        - (and more...)
        """)
    
    with col2:
        st.markdown("""
        **Categorical Features (Encoded)**
        - Gender (One-Hot)
        - Contract Length (Ordinal)
        - Subscription Type (Ordinal)
        - Internet Service (One-Hot)
        - (and more...)
        """)
    
    st.write("---")
    
    st.subheader("🎯 Target Variable Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Class Distribution (After Cleaning)**
        
        | Class | Count | Percentage |
        |-------|-------|-----------|
        | No Churn (0) | ~250K | ~82% |
        | Churn (1) | ~54K | ~18% |
        
        **Note:** Imbalanced classes handled with:
        - `scale_pos_weight` in training
        - `class_weight='balanced'` in Neural Network
        """)
    
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=['No Churn', 'Churn'],
            values=[82, 18],
            marker=dict(colors=['#1f77b4', '#ff7f0e'])
        )])
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    st.info("""
    **Note:** This dataset was preprocessed and cleaned before being used to train the ANN model. 
    All categorical features were encoded, numeric features were scaled, and outliers were removed 
    to ensure optimal model performance.
    """)
