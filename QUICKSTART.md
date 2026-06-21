# 🚀 Quick Start Guide

## 5-Minute Setup

### Step 1: Download the app folder
The entire `streamlit_app/` folder is your project.

### Step 2: Install Dependencies
```bash
cd streamlit_app
pip install -r requirements.txt
```

### Step 3: Add Your Model Files
Copy these files from your Jupyter notebook output to `models/` folder:
- `ann_model.h5`
- `final_pipeline.pkl`
- `model_config.json`

### Step 4: Run the App
```bash
streamlit run app.py
```

✅ Done! Your app is running at `http://localhost:8501`

---

## 🎯 What Each Page Does

| Page | Purpose |
|------|---------|
| 🏠 Home | Overview and welcome |
| 📈 Data Analysis | Dataset exploration & preprocessing |
| 🧠 Model Architecture | ANN explanation (layers, neurons, activation) |
| ⚖️ ANN vs ML | Compare with Random Forest from MP1 |
| 🔮 Prediction | Make churn predictions interactively |

---

## 📁 Folder Guide

```
models/              ← Put your saved model files here
pages/              ← Page modules (don't edit unless customizing)
utils/              ← Utility functions (don't edit unless customizing)
app.py              ← Main app file (run this!)
requirements.txt    ← Dependencies (pip install -r requirements.txt)
README.md          ← Full documentation
```

---

## 💡 Tips

1. **First time?** → Start with the Home page
2. **Want to understand the model?** → Go to "Model Architecture"
3. **Need to compare models?** → Visit "ANN vs ML Comparison"
4. **Making predictions?** → Use "Make Predictions" page
5. **Exploring data?** → Check "Data Analysis"

---

## 🔧 Troubleshooting

**"No such file or directory: ann_model.h5"**
→ Make sure the model files are in the `models/` folder

**"ModuleNotFoundError: No module named 'tensorflow'"**
→ Run: `pip install -r requirements.txt`

**Port 8501 already in use**
→ Run: `streamlit run app.py --server.port 8502`

---

## 📞 Need Help?

Check the full `README.md` file for detailed documentation!
