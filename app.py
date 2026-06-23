import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none;
}

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

# =========================
# Sidebar Navigation
# =========================

st.sidebar.title("🔍 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📈 Data Analysis",
        "🧠 Model Architecture",
        "⚖️ ANN vs ML Comparison",
        "🔮 Make Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Mini Project 2 - Model ANN**

    Prediksi Churn Pelanggan menggunakan Artificial Neural Network
    """
)

# =========================
# Page Routing
# =========================

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