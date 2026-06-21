import streamlit as st
import os


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📈 Data Analysis", "🧠 Model Architecture", "⚖️ ANN vs ML Comparison", "🔮 Make Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Mini Project 2 - ANN Model**\n\n"
    "Customer Churn Prediction using Artificial Neural Network"
)


if page == "🏠 Home":
    from pages import home
    home.run()
elif page == "📈 Data Analysis":
    from pages import data_analysis
    data_analysis.run()
elif page == "🧠 Model Architecture":
    from pages import model_architecture
    model_architecture.run()
elif page == "⚖️ ANN vs ML Comparison":
    from pages import model_comparison
    model_comparison.run()
elif page == "🔮 Make Prediction":
    from pages import prediction
    prediction.run()
