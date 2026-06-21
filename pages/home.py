import streamlit as st

def run():
    st.title("🏠 Customer Churn Prediction")
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to Mini Project 2! 📚
        
        This application demonstrates the development and evaluation of an **Artificial Neural Network (ANN)** 
        for predicting customer churn in a telecom dataset.
        
        #### What is This Project About?
        
        The primary focus of **Mini Project 2** is to:
        - ✅ Build and understand how ANN models work
        - ✅ Explain the model architecture in detail (layers, neurons, activation functions)
        - ✅ Compare ANN performance with Machine Learning models (from Mini Project 1)
        - ✅ Provide a user-friendly interface to make predictions
        
        #### Key Features
        
        **1. 📈 Data Analysis**
        - Explore the customer churn dataset
        - Visualize feature distributions and correlations
        - Understand data preprocessing steps
        
        **2. 🧠 Model Architecture**
        - Learn about the ANN structure
        - Understand activation functions and layers
        - See training history and model performance
        
        **3. ⚖️ Model Comparison**
        - Compare ANN vs Random Forest (ML model)
        - Analyze strengths and weaknesses
        - Review evaluation metrics
        
        **4. 🔮 Make Predictions**
        - Input customer data
        - Get churn probability predictions
        - See prediction confidence
        """)
    
    with col2:
        st.markdown("""
        ### Quick Stats 📊
        
        **Model Performance**
        """)
        st.metric("Accuracy", "98.0%", "+2.0%")
        st.metric("Precision", "100.0%", "✓")
        st.metric("Recall", "96.5%", "✓")
        st.metric("F1-Score", "98.0%", "✓")
    
    st.write("---")
    
    st.markdown("""
    ### How to Navigate
    
    Use the **sidebar** to select different sections:
    - 📈 **Data Analysis** - Explore the dataset
    - 🧠 **Model Architecture** - Understand the ANN model
    - ⚖️ **ANN vs ML Comparison** - Compare models
    - 🔮 **Make Prediction** - Try predictions
    
    ---
    
    **Created by:** Yonda Eko Dirman | **Project:** Mini Project 2 - ANN Model
    """)
