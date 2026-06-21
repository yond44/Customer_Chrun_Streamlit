import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def run():
    st.title("⚖️ ANN vs ML Model Comparison")
    st.write("---")
    
    st.markdown("""
    ### Comparing ANN with Random Forest (Mini Project 1)
    
    This section provides a detailed analysis of the two models used in this project:
    - **ANN (Artificial Neural Network)** - Mini Project 2 (current)
    - **Random Forest (ML Model)** - Mini Project 1
    """)
    
    st.write("---")
    
    # Metrics Comparison
    st.subheader("📊 Performance Metrics Comparison")
    
    # Create comparison data
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'ANN': [0.9800, 0.9809, 0.9798, 0.9799],
        'Random Forest': [0.9998, 1.0000, 1.0000, 1.0000]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(name='ANN', x=df_comparison['Metric'], y=df_comparison['ANN']),
            go.Bar(name='Random Forest', x=df_comparison['Metric'], y=df_comparison['Random Forest'])
        ])
        fig.update_layout(
            barmode='group',
            height=400,
            title='Model Metrics Comparison',
            yaxis_title='Score',
            xaxis_title='Metric'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(df_comparison.set_index('Metric'), use_container_width=True)
    
    st.write("---")
    
    # Detailed Comparison
    st.subheader("🔍 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Strengths", "Weaknesses", "Architecture", "Use Cases"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ANN Strengths ✅
            
            - **Non-linear relationships**
              - Captures complex, non-linear patterns
              - Good for intricate customer behaviors
            
            - **Scalability**
              - Can handle large datasets efficiently
              - Parallelizable architecture
            
            - **Continuous learning**
              - Can be fine-tuned with new data
              - Incremental improvement possible
            
            - **Deep understanding**
              - Learn hierarchical representations
              - Better generalization sometimes
            """)
        
        with col2:
            st.markdown("""
            #### Random Forest Strengths ✅
            
            - **Interpretability**
              - Feature importance easily extracted
              - Decision paths understandable
            
            - **Perfect performance (on this dataset)**
              - 100% accuracy achieved
              - No overfitting despite high accuracy
            
            - **Robustness**
              - Handles outliers well
              - Less sensitive to hyperparameters
            
            - **Ensemble advantage**
              - Combines multiple weak learners
              - Reduces variance effectively
            """)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ANN Weaknesses ⚠️
            
            - **Black box nature**
              - Difficult to explain predictions
              - Hard to extract feature importance
            
            - **Training time**
              - More epochs needed for convergence
              - Requires GPU for large datasets
            
            - **Hyperparameter tuning**
              - Many parameters to tune
              - More complex than tree models
            
            - **Lower performance (here)**
              - 98% vs 100% accuracy
              - Slightly higher error rate
            """)
        
        with col2:
            st.markdown("""
            #### Random Forest Weaknesses ⚠️
            
            - **Limited scalability**
              - Memory intensive for huge datasets
              - Slower prediction on new data
            
            - **Feature interactions**
              - Not as good at learning complex interactions
              - May miss some patterns
            
            - **Overfitting risk**
              - Deep trees can memorize data
              - Although regularization helps
            
            - **Update difficulty**
              - Cannot incrementally learn new data
              - Requires retraining from scratch
            """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ANN Architecture
            
            **Structure:**
            - Input: 37 features (scaled)
            - Hidden 1: 64 neurons (ReLU)
            - Hidden 2: 32 neurons (ReLU)
            - Hidden 3: 16 neurons (ReLU)
            - Output: 1 neuron (Sigmoid)
            
            **Learning Method:**
            - Gradient descent (backpropagation)
            - Weights updated iteratively
            - Non-linear transformations
            
            **Parameters:**
            - ~3,500 weights to learn
            - Adam optimizer
            - Binary crossentropy loss
            """)
        
        with col2:
            st.markdown("""
            #### Random Forest Architecture
            
            **Structure:**
            - Ensemble of Decision Trees
            - Each tree trained on data subset
            - Trees vote on final prediction
            
            **Learning Method:**
            - Information gain/Gini impurity
            - Greedy tree construction
            - Bootstrap aggregating (bagging)
            
            **Parameters:**
            - Thousands of tree nodes
              (depends on tree depth)
            - Feature splits at each node
            - Ensemble voting mechanism
            """)
    
    with tab4:
        st.markdown("""
        #### When to Use ANN?
        
        - ✅ Large datasets with complex patterns
        - ✅ Need incremental learning capability
        - ✅ Real-time predictions critical
        - ✅ Transfer learning required
        - ✅ Production scalability important
        
        #### When to Use Random Forest?
        
        - ✅ Need interpretable decisions
        - ✅ Feature importance critical
        - ✅ Dataset is medium-sized
        - ✅ Robustness > small accuracy gains
        - ✅ Limited computational resources
        
        #### For This Project
        
        **Random Forest won** because:
        - 100% test accuracy vs 98%
        - Better business interpretability
        - Feature importance insights
        - Slightly more stable
        
        **However, ANN is competitive** and demonstrates:
        - Deep learning capability
        - Neural network understanding
        - 98% accuracy still excellent
        """)
    
    st.write("---")
    
    # Summary Comparison Table
    st.subheader("📋 Quick Comparison Table")
    
    comparison_table = pd.DataFrame({
        'Aspect': [
            'Accuracy',
            'Interpretability',
            'Training Time',
            'Prediction Speed',
            'Scalability',
            'Memory Usage',
            'Hyperparameter Tuning',
            'Robustness',
            'Feature Importance'
        ],
        'ANN': [
            '98.0%',
            '⭐⭐☆☆☆ (Low)',
            '⭐⭐⭐☆☆ (Medium)',
            '⭐⭐⭐⭐⭐ (Fast)',
            '⭐⭐⭐⭐⭐ (Excellent)',
            '⭐⭐⭐⭐☆ (Good)',
            '⭐⭐☆☆☆ (Complex)',
            '⭐⭐⭐⭐☆',
            '⭐⭐☆☆☆ (Limited)'
        ],
        'Random Forest': [
            '100.0%',
            '⭐⭐⭐⭐☆ (High)',
            '⭐⭐⭐⭐☆ (Fast)',
            '⭐⭐⭐☆☆ (Slower)',
            '⭐⭐⭐☆☆ (Good)',
            '⭐⭐⭐⭐☆ (Better)',
            '⭐⭐⭐⭐⭐ (Simple)',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐ (Excellent)'
        ]
    })
    
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    
    st.write("---")
    
    st.success("""
    **Conclusion:** Both models perform exceptionally well. Random Forest achieves 100% accuracy 
    with better interpretability, while ANN achieves 98% accuracy with better scalability. 
    The choice depends on project priorities: interpretability vs scalability.
    """)
