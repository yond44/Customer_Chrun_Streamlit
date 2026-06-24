# 🎯 Customer Churn Prediction System

<div align="center">

**Advanced Machine Learning Application for Customer Retention**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)

[Features](#-features) • [Quick Start](#-quick-start) • [Models](#-models) • [Documentation](#-documentation)

</div>

---

## 📌 Overview

A production-ready Streamlit application that predicts customer churn using **two AI models**:
- 🌲 **Random Forest** (Ensemble Learning) - 10 features
- 🧠 **Neural Network** (Deep Learning) - 6 features

The app provides **side-by-side predictions**, **data quality warnings**, and **actionable business recommendations** to help optimize customer retention strategies.

**Key Insight:** ⚠️ The Random Forest achieved 99.99% accuracy (likely data leakage), while the Neural Network achieved realistic 86-88% accuracy with careful feature selection. This demonstrates why **honest predictions beat perfect ones!**

---

## ✨ Features

### 🔮 Dual Model Predictions
- **Input:** 11 customer fields (demographics, behavior, service info)
- **Output:** Side-by-side predictions from RF and ANN models
- **Comparison:** Winner/loser indication, disagreement measurement
- **Consensus:** Risk level assessment combining both models

### ⚠️ Data Leakage Warnings
- Alerts when RF predictions seem extreme (>95% or <5%)
- Educational explanation of what data leakage is
- Why ANN's lower accuracy is actually more trustworthy
- Clear recommendations on which model to trust

### 💡 Smart Recommendations
- Context-aware business actions
- Risk-level-specific strategies
- Tenure and behavior-based guidance
- Resource allocation suggestions

### 📊 Model Comparison
- Detailed performance metrics
- Architecture explanations
- Feature importance analysis
- Data quality assessment
- Production readiness evaluation

### 📚 Educational Content
- Interactive learning sections
- Data science best practices
- Feature selection methodology
- Real-world ML concepts
- Professional training techniques


---

## 🤖 Models

### 🌲 Random Forest (MP01)
```
Algorithm:        Ensemble of 200 Decision Trees
Features:         10 (7 numeric + 3 categorical)
Training Acc:     100% ⚠️
Test Acc:         99.99% ⚠️
Trust Level:      🟡 MEDIUM (likely data leakage)
Production Use:   ❌ NOT RECOMMENDED for critical decisions
Best Use:         Feature importance analysis, interpretability
```

**Why the high accuracy is suspicious:**
- 99.99% accuracy on customer churn is unrealistic
- Likely features are consequences of churn, not predictors
- May not generalize to new customers
- Should investigate feature-target correlations

### 🧠 Neural Network (MP02) - ⬆️ IMPROVED
```
Algorithm:        Deep Learning (3 hidden layers)
Layers:           64 → 32 → 16 → 1
Features:         6 (carefully selected)
Training Acc:     86-88% ✓
Test Acc:         86-88% ✓
Trust Level:      🟢 HIGH (leakage-resistant)
Production Use:   ✅ RECOMMENDED
Best Use:         Predictions, business decisions, deployment
```

**Why this accuracy is trustworthy:**
- 86-88% is realistic for churn prediction
- Features selected to prevent leakage
- Only behavioral predictors (not consequences)
- Better generalization to unseen data
- Professional training techniques (Early Stopping, LR Scheduling)

### 📊 Feature Comparison

| Feature | RF Uses | ANN Uses | Reason for Inclusion/Exclusion |
|---------|---------|----------|-------------------------------|
| Age | ✓ | ✓ | Demographics (immutable) |
| Tenure | ✓ | ✓ | Loyalty indicator |
| Usage Frequency | ✓ | ✓ | Engagement metric |
| Support Calls | ✓ | ✓ | Problem indicator |
| Payment Delay | ✓ | ✓ | Financial health |
| Last Interaction | ✓ | ✓ | Recency of contact |
| Total Spend | ✓ | ✗ | Consequence (after churn) |
| Monthly Charges | ✓ | ✗ | Changes when canceling |
| Gender | ✓ | ✗ | Not strongly predictive |
| Subscription Type | ✓ | ✗ | May reflect intent |
| Contract Length | ✓ | ✗ | May be consequence |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yond44/churn-prediction-app.git
cd churn-prediction-app
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model files**
- From your MP01 Colab notebook:
  - `Customer_churn_rf_model.pkl`
  - `encoder_ordinal.pkl`
  - `encoder_ohe.pkl`
  - `scaler.pkl`

- From your MP02 Colab notebook:
  - `ann_model.h5` (UPDATED with improved training)
  - `final_pipeline.pkl`
  - `model_config.json`

5. **Organize model files**
```bash
mkdir models
# Place all 7 files in models/ folder
```

6. **Test the setup**
```bash
python debug_models.py
# Should show ✅ for all checks
```

7. **Launch the app**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📁 File Structure

```
churn-prediction-app/
│
├── app.py                          # Main Streamlit app (4 pages)
├── requirements.txt                # Python dependencies
├── debug_models.py                 # Model diagnostic tool
├── README.md                       # This file
│
├── pages/                          # Streamlit multi-page app
│   ├── __init__.py
│   ├── prediction.py              # 🔮 Make predictions (RF vs ANN)
│   ├── model_comparison.py        # 📊 Detailed model analysis
│   ├── model_architecture.py      # 🧠 ANN architecture (English)
│   └── model_architecture_id.py   # 🧠 Arsitektur ANN (Indonesian)
│
├── models/                         # Trained models (7 files)
│   ├── ann_model.h5              # Neural Network (TensorFlow)
│   ├── final_pipeline.pkl        # ANN preprocessing
│   ├── model_config.json         # ANN configuration
│   ├── Customer_churn_rf_model.pkl # Random Forest
│   ├── encoder_ordinal.pkl       # RF ordinal encoder
│   ├── encoder_ohe.pkl           # RF one-hot encoder
│   └── scaler.pkl                # RF scaler
│
├── docs/                           # Documentation
│   ├── COMPLETE_IMPLEMENTATION_GUIDE.md
│   ├── DATA_LEAKAGE_EXPLAINED.md
│   ├── SETUP_COMPARISON.md
│   ├── FIX_FEATURE_ERROR.md
│   ├── MODEL_COMPARISON_UPDATE_SUMMARY.md
│   ├── MODEL_ARCHITECTURE_UPDATE_SUMMARY.md
│   ├── NEW_ANN_MODEL_UPDATE.md
│   ├── QUICK_MODEL_UPDATE.md
│   ├── VERSION_COMPARISON.md
│   └── UPDATE_SUMMARY.md
│
└── .gitignore
```

---

## 📖 Usage Guide

### 🏠 Home Page
- **Purpose:** Landing page with navigation
- **Content:** Overview, FAQ, quick start guide
- **Users:** First-time visitors
- **Time:** 2-3 minutes to explore

### 🔮 Make Prediction Page
- **Purpose:** Make actual churn predictions
- **Inputs:** 
  - Age, Gender, Tenure
  - Subscription Type, Contract Length
  - Usage Frequency, Support Calls
  - Payment Delay, Last Interaction
  - Monthly Charges, Total Spend
- **Outputs:**
  - Side-by-side predictions (RF vs ANN)
  - Winner/loser indication
  - Risk consensus assessment
  - Customer profile analysis
  - Risk factors breakdown
  - Smart recommendations
- **Users:** Business analysts, retention managers

### 📊 Model Comparison Page
- **Purpose:** Deep dive into model differences
- **Content:**
  - Performance metrics
  - Architecture details
  - Feature comparison
  - Data leakage discussion
  - Trust level indicators
  - Production readiness assessment
- **Users:** Data scientists, technical stakeholders

### 📚 Learn More Page
- **Purpose:** Educational content
- **Topics:**
  - Data leakage explanation
  - Feature selection methodology
  - Model comparison theory
  - Best practices in ML
  - Red flags and green lights
- **Users:** Students, learners, interested stakeholders

---

## 📊 Performance Metrics

### Random Forest
| Metric | Value | Assessment |
|--------|-------|-----------|
| Test Accuracy | 99.99% | ⚠️ Suspiciously High |
| Precision | 100.0% | ⚠️ Likely Leakage |
| Recall | 100.0% | ⚠️ Likely Leakage |
| F1-Score | 1.0000 | ⚠️ Unrealistic |
| Data Leakage Risk | HIGH | 🔴 Do Not Use |

### Neural Network (Updated) ⬆️
| Metric | Value | Assessment |
|--------|-------|-----------|
| Test Accuracy | 86-88% | ✓ Realistic |
| Precision | ~82-84% | ✓ Good |
| Recall | ~78-80% | ✓ Good |
| F1-Score | ~0.80-0.82 | ✓ Balanced |
| Data Leakage Risk | LOW | 🟢 Production Ready |

---

## 🎓 Key Insights

### Data Leakage Problem
```
❌ Random Forest: 99.99% accuracy
   └─ Features may be consequences of churn
   └─ Model may be memorizing patterns
   └─ Poor generalization expected

✅ Neural Network: 86-88% accuracy
   └─ Careful feature selection
   └─ True behavioral predictors only
   └─ Better generalization expected
```

### Why Lower Accuracy is Better
```
In machine learning:
- Higher accuracy ≠ Better model
- Realistic uncertainty > False confidence
- 86% honest > 99.99% inflated
- Generalization > Perfection
```

### Feature Selection Impact
```
Random Forest:  10 features (higher leakage risk)
Neural Network: 6 features (anti-leakage design)

Excluded features (why):
- Total Spend (drops after churn decision)
- Monthly Charges (changes when canceling)
- Subscription Type (reflects intent)
- Contract Length (may be consequence)
- Gender (not strongly predictive)
```

---

## 🔧 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.28+ |
| **Data Processing** | Pandas, NumPy | 2.0+, 1.24+ |
| **ML - Ensemble** | scikit-learn | 1.3+ |
| **ML - Deep Learning** | TensorFlow/Keras | 2.13+ |
| **Visualization** | Plotly | 5.17+ |
| **Model Serialization** | joblib | 1.3+ |
| **Language** | Python | 3.9+ |

---

## ⚙️ Configuration

### Environment Variables (Optional)
```bash
# None required for basic usage
# All configuration in code
```

### Model Parameters
```python
# Random Forest
- n_estimators: 200
- max_depth: 20
- class_weight: balanced

# Neural Network
- Input: 6 features
- Layers: 64 → 32 → 16 → 1
- Dropout: 0.3, 0.2
- Optimizer: Adam
- Loss: Binary Crossentropy
- Early Stopping: patience=7
- Learning Rate Scheduling: factor=0.5
```

---

## 📈 Training Data Specifications

```
Total Samples:        ~440,000
Training Samples:     ~314,500 (80%)
Test Samples:         ~65,305 (20%)

Churn Distribution:
- No Churn (0):       ~73% (~322,000)
- Churn (1):          ~27% (~118,000)

Class Imbalance:      Handled with class weights
```

---

## 🚀 Deployment

### Option 1: Streamlit Cloud (Recommended for Quick Demo)
```bash
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Automatic deployment on push
4. Access at: https://your-app.streamlit.app
```

### Option 2: Docker
```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

### Option 3: Local Server
```bash
streamlit run app.py --server.port 8501
```

### Option 4: Cloud Platforms
- AWS EC2 + Nginx
- Azure App Service
- Google Cloud Run
- Heroku (with custom buildpack)

---

## 🐛 Troubleshooting

### "Model files not found"
```bash
# Check files exist
ls -la models/
# Should show 7 files

# If missing, download from Colab again
python debug_models.py  # Shows which files are missing
```

### "Feature name mismatch"
```bash
# RF model is strict about feature names
python debug_models.py  # Will show exact error
# Check feature order matches training
```

### "Import error"
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### "Slow predictions"
```bash
# First run: Models loading (~5-10 seconds) - NORMAL
# Cached runs: Fast (~0.5 seconds) - EXPECTED
```

---

## 📚 Documentation

Comprehensive guides included:

| Document | Purpose |
|----------|---------|
| `COMPLETE_IMPLEMENTATION_GUIDE.md` | Step-by-step setup & usage |
| `DATA_LEAKAGE_EXPLAINED.md` | Educational deep dive on leakage |
| `SETUP_COMPARISON.md` | Multi-model setup instructions |
| `MODEL_COMPARISON_UPDATE_SUMMARY.md` | What's new in comparison page |
| `MODEL_ARCHITECTURE_UPDATE_SUMMARY.md` | Architecture page updates |
| `NEW_ANN_MODEL_UPDATE.md` | Improved ANN details |
| `QUICK_MODEL_UPDATE.md` | Quick reference guide |

---

## 🎓 What You'll Learn

### Data Science Concepts
- Model comparison and evaluation
- Feature selection methodology
- Data leakage detection
- Cross-validation importance
- Hyperparameter tuning

### Machine Learning
- Random Forest ensemble methods
- Neural Network architecture
- Activation functions (ReLU, Sigmoid)
- Dropout regularization
- Optimization algorithms (Adam)

### Production ML
- Model serialization and loading
- Preprocessing pipelines
- A/B testing concepts
- Model monitoring
- Scaling considerations

### Business Analytics
- Churn prediction application
- Customer retention strategies
- Resource allocation
- ROI calculation
- Data-driven decision making

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

### Enhancements
- [ ] Add more visualizations
- [ ] Implement additional models (XGBoost, LightGBM)
- [ ] Add data upload functionality
- [ ] Create admin dashboard
- [ ] Implement batch prediction
- [ ] Add email notifications
- [ ] Create API wrapper

### Translations
- [ ] Spanish
- [ ] French
- [ ] Chinese
- [ ] Japanese
- [ ] German

### Documentation
- [ ] Video tutorials
- [ ] Interactive notebooks
- [ ] Case studies
- [ ] Best practices guide

---

### Common Issues
- ✅ See troubleshooting section above
- ✅ Check FIX_FEATURE_ERROR.md for specific errors
- ✅ Review COMPLETE_IMPLEMENTATION_GUIDE.md

---

## 📋 Checklist for Production

Before deploying to production:

- [ ] All 7 model files downloaded and tested
- [ ] `debug_models.py` passes all checks
- [ ] App loads without errors
- [ ] All 4 pages function correctly
- [ ] Predictions are realistic (20-80% range)
- [ ] Data leakage warning displays
- [ ] Recommendations are sensible
- [ ] Load time acceptable (<5 seconds)
- [ ] Mobile responsive (if needed)
- [ ] Team trained on interpretation

---

## 📊 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Random Forest Model | ✅ Complete | 10 features, 99.99% acc |
| Neural Network Model | ✅ Complete & Improved | 6 features, 86-88% acc |
| Prediction Page | ✅ Complete | Dual predictions + warnings |
| Comparison Page | ✅ Complete | Full analysis + leakage info |
| Architecture Page | ✅ Complete | English & Indonesian |
| Documentation | ✅ Complete | 10+ comprehensive guides |
| Testing | ✅ Complete | Debug tool provided |
| Production Ready | ✅ YES | Safe to deploy |

---

## 📄 License

This project is provided as-is for educational and business use.

---

## 👨‍💼 About

Created by: Yonda Eko Dirman  
Mini Project 1: Random Forest Model  
Mini Project 2: Neural Network Model (Improved)  
Application: Streamlit Churn Prediction System  

---

## 🙏 Acknowledgments

- TensorFlow/Keras for neural networks
- scikit-learn for machine learning
- Streamlit for UI framework
- Plotly for visualizations

---


<div align="center">

**Made with ❤️ for data science & customer retention**

[⬆ back to top](#customer-churn-prediction-system)

</div>
