import streamlit as st
import plotly.graph_objects as go

def run():
    st.title("🧠 ANN Model Architecture")
    st.write("---")
    
    st.markdown("""
    ### Understanding the Artificial Neural Network
    
    This section explains the architecture, design decisions, and performance of the ANN model 
    built specifically for customer churn prediction.
    """)
    
    st.write("---")
    
    st.subheader("🏗️ Network Architecture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Layer Configuration**
        
        | Layer | Type | Units | Activation | Dropout |
        |-------|------|-------|-----------|---------|
        | Input | Input | 37 | - | - |
        | Hidden 1 | Dense | 64 | ReLU | 0.3 |
        | Hidden 2 | Dense | 32 | ReLU | 0.2 |
        | Hidden 3 | Dense | 16 | ReLU | - |
        | Output | Dense | 1 | Sigmoid | - |
        
        **Total Parameters:** ~3,500+
        """)
    
    with col2:
        st.metric("Input Features", "37")
        st.metric("Hidden Layers", "3")
        st.metric("Total Layers", "5")
    
    st.markdown("""
    #### Network Flow Diagram
    
    ```
    ┌─────────────────────────────────────────────────┐
    │  INPUT LAYER (37 features)                      │
    │  Scaled numeric + Encoded categorical           │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │  HIDDEN LAYER 1: 64 neurons                     │
    │  Activation: ReLU (Rectified Linear Unit)       │
    │  Dropout: 0.3 (30% regularization)              │
    │  Purpose: Learn complex patterns                │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │  HIDDEN LAYER 2: 32 neurons                     │
    │  Activation: ReLU                               │
    │  Dropout: 0.2 (20% regularization)              │
    │  Purpose: Refine learned representations        │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │  HIDDEN LAYER 3: 16 neurons                     │
    │  Activation: ReLU                               │
    │  Purpose: Final feature transformation          │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │  OUTPUT LAYER: 1 neuron                         │
    │  Activation: Sigmoid (0-1 probability)          │
    │  Output: Churn probability                      │
    └─────────────────────────────────────────────────┘
    ```
    """)
    
    st.write("---")
    
    st.subheader("⚡ Activation Functions Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ReLU (Rectified Linear Unit)**
        
        Used in hidden layers.
        
        ```
        f(x) = max(0, x)
        ```
        
        **Why ReLU?**
        - Introduces non-linearity
        - Computationally efficient
        - Avoids vanishing gradient problem
        - Allows network to learn complex patterns
        """)
    
    with col2:
        st.markdown("""
        **Sigmoid**
        
        Used in output layer.
        
        ```
        f(x) = 1 / (1 + e^-x)
        ```
        
        **Why Sigmoid?**
        - Outputs probability (0-1)
        - Perfect for binary classification
        - Interprets as "probability of churn"
        """)
    
    st.write("---")
    
    st.subheader("🎓 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Optimization & Loss**
        
        - **Optimizer:** Adam
          - Adaptive learning rate
          - Efficient convergence
        
        - **Loss Function:** Binary Crossentropy
          - Standard for binary classification
          - Measures prediction error
        
        - **Metrics:**
          - Accuracy
          - Precision
          - Recall
        """)
    
    with col2:
        st.markdown("""
        **Regularization & Callbacks**
        
        - **Dropout:** 0.3, 0.2
          - Prevents overfitting
          - Reduces co-adaptation
        
        - **Early Stopping:**
          - Monitor validation loss
          - Patience: 7 epochs
        
        - **Learning Rate Scheduler:**
          - Reduce on plateau
          - Factor: 0.5
        """)
    
    st.write("---")
    
    st.subheader("📊 Model Performance")
    
    tab1, tab2, tab3 = st.tabs(["Training Results", "Test Results", "Confusion Matrix"])
    
    with tab1:
        st.markdown("""
        **Classification Report - Training Set**
        
        | Metric | Class 0 | Class 1 | Weighted Avg |
        |--------|---------|---------|------------|
        | Precision | 0.9558 | 1.0000 | 0.9809 |
        | Recall | 1.0000 | 0.9647 | 0.9800 |
        | F1-Score | 0.9774 | 0.9820 | 0.9797 |
        | Support | 136,249 | 178,298 | 314,547 |
        
        **Accuracy:** 98.00%
        """)
    
    with tab2:
        st.markdown("""
        **Classification Report - Test Set**
        
        | Metric | Class 0 | Class 1 | Weighted Avg |
        |--------|---------|---------|------------|
        | Precision | 0.9554 | 0.9999 | 0.9807 |
        | Recall | 0.9999 | 0.9646 | 0.9798 |
        | F1-Score | 0.9772 | 0.9820 | 0.9799 |
        | Support | 28,152 | 37,153 | 65,305 |
        
        **Accuracy:** 97.98%
        
        ✅ **Insight:** Train and test performance are very similar → No overfitting!
        """)
    
    with tab3:
        st.markdown("""
        **Confusion Matrix - Test Set**
        
        ```
        Actual / Predicted    |    No Churn    |    Churn
        ─────────────────────────────────────────────────
        No Churn (0)          |     28,150     |      2
        Churn (1)             |        309     | 35,839
        ─────────────────────────────────────────────────
        ```
        
        **Interpretation:**
        - **True Negatives (TN):** 28,150 - Correctly predicted no churn
        - **False Positives (FP):** 2 - Incorrectly predicted churn
        - **False Negatives (FN):** 309 - Missed churn cases
        - **True Positives (TP):** 35,839 - Correctly predicted churn
        """)
    
    st.write("---")
    
    st.subheader("🎯 Design Decisions & Rationale")
    
    st.markdown("""
    1. **Why 3 Hidden Layers?**
       - Complex patterns in customer behavior require multiple layers
       - Progressive feature refinement (64 → 32 → 16 neurons)
       - Reduces dimensionality gracefully toward output
    
    2. **Why Dropout?**
       - Prevents overfitting by randomly disabling neurons
       - Forces network to learn redundant representations
       - Improves generalization to unseen data
    
    3. **Why Class Weights?**
       - Dataset is imbalanced (82% vs 18% churn)
       - Balanced weights prevent bias toward majority class
       - Ensures model learns both classes equally well
    
    4. **Why These Metrics?**
       - **Accuracy:** Overall correctness
       - **Precision:** When model predicts churn, is it right?
       - **Recall:** What % of actual churners are caught?
       - **F1-Score:** Harmonic mean (balanced view)
    """)
    
    st.write("---")
    
    st.info("""
    **Key Takeaway:** This ANN architecture balances complexity (enough layers for patterns) 
    with simplicity (not too deep = fast training, interpretable). The 98% accuracy demonstrates 
    that the model successfully learns customer churn patterns.
    """)
