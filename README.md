# Personal Finance Classification Using Machine Learning
> **B.Tech CSE Semester V — Machine Learning Case Study 135**  
> An end-to-end Machine Learning system trained on the real-world **Indian Personal Finance and Spending Habits Dataset** (20,000 consumer records from Kaggle/GitHub), classifying individuals into **Saver**, **Balanced**, and **High-Spender** behavioral archetypes, providing automated financial health scoring, and delivering actionable advisory recommendations.

---

## 🌟 Features
- **Real-World Online Dataset:** Sourced directly from [Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits) & [GitHub](https://github.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits) containing **20,000 real individual financial profiles**.
- **6 Supervised Machine Learning Algorithms Benchmarked:**
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Decision Tree
  4. Random Forest
  5. Gradient Boosting
  6. Support Vector Machine (SVM)
- **High Predictive Performance:** Up to **99.98% Test Accuracy** and **99.98% Macro F1-Score**.
- **Comprehensive Exploratory Data Analysis (EDA):** 6 statistical visualizations uncovering spending patterns, correlation dynamics, and archetype clustering.
- **Modern Interactive Web Application:** Built with Flask, bespoke CSS design system, and dynamic Chart.js visualizations (spending breakdown & 50/30/20 target budget simulator).
- **Automated Financial Advisory:** Calculates financial health scores (0-100), detects vulnerability triggers (high DTI, credit card trap), and serves tailored recommendations.
- **REST API:** Production-ready endpoints for programmatic batch and real-time predictions.
- **Jupyter Notebook Included:** Fully runnable `notebooks/Personal_Finance_Classification.ipynb` with complete markdown explanations.

---

## 📁 Repository Structure
```
.
├── app.py                                   # Flask web server & REST API
├── data/
│   ├── load_online_dataset.py               # Ingestion script for real Kaggle/GitHub data
│   ├── personal_finance_data.csv            # Standardized 20,000-record dataset
│   ├── dataset_source.json                  # Provenance metadata & source URLs
│   └── eda_summary.json                     # Precomputed archetype statistics
├── models/
│   ├── decision_tree_model.joblib           # Trained Decision Tree model (Best Model)
│   ├── gradient_boosting_model.joblib       # Trained Gradient Boosting model
│   ├── k_nearest_neighbors_model.joblib     # Trained KNN model
│   ├── logistic_regression_model.joblib     # Trained Logistic Regression model
│   ├── random_forest_model.joblib           # Trained Random Forest model
│   ├── support_vector_machine_model.joblib  # Trained SVM model
│   ├── scaler.joblib                        # Fitted StandardScaler
│   ├── label_encoder.joblib                 # Fitted LabelEncoder
│   └── metrics.json                         # Full comparative evaluation metrics
├── notebooks/
│   ├── Personal_Finance_Classification.ipynb # Complete Jupyter Case Study Notebook
│   └── generate_notebook.py                 # Notebook generation utility
├── reports/
│   └── figures/                             # Exported high-res evaluation plots
├── src/
│   ├── eda.py                               # Exploratory data analysis & figure generator
│   ├── train_models.py                      # Multi-model training & comparative study
│   └── predictor.py                         # Inference engine & financial advisory
├── static/
│   ├── css/style.css                        # Modern responsive dark-mode styling
│   ├── js/main.js                           # Frontend interaction & dynamic charts
│   └── plots/                               # Static plots served on the web dashboard
├── templates/
│   ├── base.html                            # Glassmorphic layout wrapper
│   ├── index.html                           # Live prediction studio & financial health
│   ├── comparison.html                      # Comparative study of 6 ML models
│   ├── eda.html                             # Exploratory data analysis & patterns
│   └── report.html                          # Academic final analysis report
├── PROJECT_REPORT.md                        # Formal academic project documentation
└── README.md                                # Project documentation & setup guide
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed along with standard scientific libraries:
```bash
python3 -m pip install flask scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Launch the Web Application
Start the local server:
```bash
python3 app.py
```
Open your browser and navigate to:
```
http://localhost:5001
```

---

## 📊 Re-running the Machine Learning Pipeline

### Step 1: Ingest Real Online Dataset
```bash
python3 data/load_online_dataset.py
```
Fetches the raw 20,000-row dataset from GitHub/Kaggle and standardizes features.

### Step 2: Run Exploratory Data Analysis (EDA)
```bash
python3 src/eda.py
```
Generates 6 statistical visualization figures in `reports/figures/` and `static/plots/`.

### Step 3: Train & Benchmark All 6 Models
```bash
python3 src/train_models.py
```
Trains the 6 algorithms with 5-fold stratified cross validation, computes evaluation metrics, exports confusion matrices, and persists models into `models/`.

---

## 🔗 REST API Endpoints

### 1. Predict Financial Category
`POST /api/predict`

**Request Payload:**
```json
{
  "monthly_income": 48000,
  "monthly_expenses": 28000,
  "savings": 20000,
  "loan_payments": 2000,
  "investment_amount": 2500,
  "housing_utilities": 12000,
  "food_dining": 8000,
  "transportation": 3000,
  "healthcare": 2000,
  "entertainment": 1500,
  "shopping_discretionary": 1500,
  "credit_card_utilization": 25.0,
  "model_name": "Decision Tree"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predicted_category": "Saver",
    "confidence": 100.0,
    "health_score": 93,
    "model_used": "Decision Tree",
    "probabilities": {
      "Balanced": 0.0,
      "High-Spender": 0.0,
      "Saver": 100.0
    },
    "metrics": {
      "savings_rate": 41.67,
      "expense_to_income": 58.33,
      "debt_to_income": 4.17,
      "discretionary_ratio": 10.71
    },
    "recommendations": [...],
    "alerts": []
  }
}
```

### 2. Retrieve Model Comparison Metrics
`GET /api/metrics`

### 3. Retrieve EDA Summary Statistics
`GET /api/eda-summary`
