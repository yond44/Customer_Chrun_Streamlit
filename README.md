# 🧠 Customer Churn Prediction - ANN Model Streamlit App

A modular Streamlit application for predicting customer churn using an Artificial Neural Network (ANN) model. This project is **Mini Project 2**, comparing ANN with Random Forest from Mini Project 1.

## 📁 Project Structure

```
streamlit_app/
├── app.py                      # Main Streamlit app entry point
├── models/                     # Saved model files
│   ├── ann_model.h5           # Trained ANN model
│   ├── final_pipeline.pkl     # Preprocessing pipeline
│   └── model_config.json      # Model configuration
├── pages/                      # Page modules
│   ├── __init__.py
│   ├── home.py                # Home/Welcome page
│   ├── data_analysis.py       # Dataset exploration
│   ├── model_architecture.py  # ANN architecture explanation
│   ├── model_comparison.py    # ANN vs Random Forest
│   └── prediction.py          # Make predictions
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── model_utils.py         # Model loading, preprocessing
├── data/                       # Data files (if needed)
├── assets/                     # Images, styles, etc.
└── README.md                   # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install required packages
pip install streamlit pandas numpy tensorflow keras joblib plotly scikit-learn
```

### 2. Save Your Model (from Jupyter notebook)

After training your ANN model in the notebook, run these cells:

```python
# Save the model
ann_model.save('ann_model.h5')

# Save preprocessing pipeline
joblib.dump(final_pipeline, 'final_pipeline.pkl')

# Save configuration
model_config = {
    'num_columns': num_columns,
    'ordinal_categories': ordinal_categories,
    'input_dim': input_dim
}
with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
```

### 3. Organize Files

Place the three saved files in the `models/` folder:

```
streamlit_app/
└── models/
    ├── ann_model.h5
    ├── final_pipeline.pkl
    └── model_config.json
```

### 4. Install Streamlit

```bash
pip install streamlit
```

### 5. Run the App

```bash
# From the streamlit_app directory
streamlit run app.py

# Or from parent directory
streamlit run streamlit_app/app.py
```

The app will open at `http://localhost:8501`

## 📄 Page Descriptions

### 🏠 Home
- Welcome and project overview
- Quick statistics about model performance
- Navigation guide

### 📈 Data Analysis
- Dataset preprocessing steps
- Missing value handling (mode imputation)
- Outlier removal (Z-score method)
- Feature encoding strategies
- Target variable distribution

### 🧠 Model Architecture
- **Detailed ANN architecture explanation**
- Layer-by-layer breakdown:
  - Input: 37 features
  - Hidden 1: 64 neurons (ReLU, Dropout 0.3)
  - Hidden 2: 32 neurons (ReLU, Dropout 0.2)
  - Hidden 3: 16 neurons (ReLU)
  - Output: 1 neuron (Sigmoid)
- Activation functions explained (ReLU, Sigmoid)
- Training configuration (Adam optimizer, Binary Crossentropy)
- Model performance metrics
- Design decisions and rationale

### ⚖️ ANN vs ML Comparison
- **Compares ANN with Random Forest (Mini Project 1)**
- Performance metrics comparison
- Strengths and weaknesses of each model
- Architecture differences
- Use case recommendations
- Summary comparison table

### 🔮 Make Predictions
- **Interactive prediction interface**
- User input form for customer features
- Real-time prediction with confidence score
- Gauge chart visualization
- Customer profile summary
- AI-powered recommendations
- Prediction interpretation

## 🛠️ Customization

### Adding Your Model Files

1. Save your trained model from Jupyter:
```python
# From your notebook
ann_model.save('models/ann_model.h5')
joblib.dump(final_pipeline, 'models/final_pipeline.pkl')
```

2. Update `app.py` to load models if using prediction page

### Modifying Styling

Edit the CSS in `app.py`:
```python
st.markdown("""
    <style>
    /* Your custom CSS here */
    </style>
""", unsafe_allow_html=True)
```

### Adding New Pages

1. Create new file in `pages/` folder:
```python
# pages/my_new_page.py
def run():
    st.title("My New Page")
    st.write("Content here...")
```

2. Add to navigation in `app.py`:
```python
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 My New Page", ...]
)

if page == "📊 My New Page":
    from pages import my_new_page
    my_new_page.run()
```

## 📊 Key Project Goals (As Per Requirements)

✅ **Focus on understanding ANN**: Model architecture page explains layers, neurons, activation functions in detail

✅ **Explain architecture explicitly**: Architecture page details:
- 3 hidden layers with specific neuron counts
- ReLU and Sigmoid activation functions
- Dropout regularization strategy

✅ **Compare ANN vs ML (Mini Project 1)**:
- Dedicated comparison page
- Analyze which is better (Random Forest: 100%, ANN: 98%)
- Discuss strengths/weaknesses (interpretability vs scalability)

✅ **Minimal Streamlit interface**:
- Clean, uncluttered design
- User input → Prediction button → Results
- No unnecessary complexity

## 🔮 Making Predictions

The prediction page includes:
1. **Input Form**: Customer information
2. **Prediction Output**: Churn probability with gauge chart
3. **Risk Assessment**: HIGH/MEDIUM/LOW churn risk
4. **Recommendations**: Actionable business insights
5. **Confidence Score**: Based on model accuracy (98%)

## 📦 Dependencies

```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
tensorflow>=2.8.0
keras>=2.8.0
joblib>=1.1.0
plotly>=5.0.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

## 🐛 Troubleshooting

**Model files not found**
- Ensure `models/` folder exists with all three files
- Check file paths in `utils/model_utils.py`

**TensorFlow/Keras errors**
- Update: `pip install --upgrade tensorflow`
- Ensure compatible versions

**Port already in use**
- Use: `streamlit run app.py --server.port 8502`

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Plotly Charts](https://plotly.com/python/)

## ✅ Checklist Before Deployment

- [ ] Model files saved and placed in `models/` folder
- [ ] All dependencies installed
- [ ] App runs without errors: `streamlit run app.py`
- [ ] All pages load correctly
- [ ] Prediction page works with sample input
- [ ] Compare ANN vs RF metrics are accurate
- [ ] Architecture explanation is clear and detailed

---

**Created by**: Yonda Eko Dirman  
**Project**: Mini Project 2 - ANN Model  
**Date**: 2024  
**Status**: ✅ Ready to Deploy
