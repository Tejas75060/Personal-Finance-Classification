# Personal Finance Classification Using Machine Learning
> **B.Tech CSE Semester V — Machine Learning Case Study 135**  
> An end-to-end Machine Learning system trained on the real-world **Indian Personal Finance and Spending Habits Dataset** (20,000 consumer records from Kaggle/GitHub), classifying individuals into **Saver**, **Balanced**, and **High-Spender** behavioral archetypes, providing automated financial health scoring, and delivering actionable advisory recommendations.

---

## 🌟 Features
- **Real-World Online Dataset:** Sourced directly from [Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits) & [GitHub](https://github.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits) containing **20,000 real individual financial profiles**.
- **4 Best Supervised Machine Learning Algorithms Benchmarked:**
  1. Decision Tree (99.98% Test Accuracy — ★ Best Model)
  2. Random Forest (99.98% Test Accuracy)
  3. Gradient Boosting (99.98% Test Accuracy)
  4. Logistic Regression (99.62% Test Accuracy)
- **High Predictive Performance:** Up to **99.98% Test Accuracy** and **99.98% Macro F1-Score**.
- **Comprehensive Exploratory Data Analysis (EDA):** 6 statistical visualizations uncovering spending patterns, correlation dynamics, and archetype clustering.
- **Interactive Streamlit Web Application:** Deployed on Streamlit Cloud and locally runnable with live classification, preset scenarios, 50/30/20 budget simulations, and comparative studies.
- **Automated Financial Advisory:** Calculates financial health scores (0-100), detects vulnerability triggers (high DTI, credit card trap), and serves tailored recommendations.
- **Jupyter Notebook Included:** Fully runnable `notebooks/Personal_Finance_Classification.ipynb` with complete markdown explanations.

---

## 📁 Repository Structure
```
.
├── streamlit_app.py                         # Streamlit Cloud & local interactive web app
├── data/
│   ├── load_online_dataset.py               # Ingestion script for real Kaggle/GitHub data
│   ├── personal_finance_data.csv            # Standardized 20,000-record dataset
│   ├── dataset_source.json                  # Provenance metadata & source URLs
│   └── eda_summary.json                     # Precomputed archetype statistics
├── models/
│   ├── decision_tree_model.joblib           # Trained Decision Tree model (Best Model)
│   ├── gradient_boosting_model.joblib       # Trained Gradient Boosting model
│   ├── logistic_regression_model.joblib     # Trained Logistic Regression model
│   ├── random_forest_model.joblib           # Trained Random Forest model
│   ├── scaler.joblib                        # Fitted StandardScaler
│   ├── label_encoder.joblib                 # Fitted LabelEncoder
│   └── metrics.json                         # Full comparative evaluation metrics
├── notebooks/
│   ├── Personal_Finance_Classification.ipynb # Complete Jupyter Case Study Notebook
│   └── generate_notebook.py                 # Notebook generation utility
├── reports/
│   └── figures/                             # Exported high-res evaluation plots & EDA figures
├── src/
│   ├── eda.py                               # Exploratory data analysis & figure generator
│   ├── train_models.py                      # Multi-model training & comparative study
│   └── predictor.py                         # Inference engine & financial advisory
├── docs/                                    # In-depth educational documentation
│   ├── 1_ML_TOPICS_EXPLAINED.md             # Guide to every ML concept (How, Why, When)
│   ├── 2_PROJECT_FILES_EXPLAINED.md         # Guide to every file in the project (How, Why, When)
│   └── 3_ML_MODELS_EXPLAINED.md             # Guide to every ML model used (How, Why, When)
├── PROJECT_REPORT.md                        # Academic project documentation
└── README.md                                # Project documentation & setup guide
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Start the interactive application locally:
```bash
streamlit run streamlit_app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```
Or access the live deployment directly at:
[https://personal-finance-classification-kgf4ksx4x4nbbg2dssj6nr.streamlit.app/](https://personal-finance-classification-kgf4ksx4x4nbbg2dssj6nr.streamlit.app/)

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

### Step 3: Train & Benchmark the 4 Best Models
```bash
python3 src/train_models.py
```
Trains the 4 best algorithms with 5-fold stratified cross validation, computes evaluation metrics, exports confusion matrices, and persists models into `models/`.

---

## 💡 Programmatic Python Inference

You can also run inference directly in Python using `FinancialClassifierService`:

```python
from src.predictor import FinancialClassifierService

service = FinancialClassifierService()

user_input = {
    'monthly_income': 48000,
    'monthly_expenses': 27000,
    'savings': 18000,
    'loan_payments': 2000,
    'investment_amount': 2500,
    'housing_utilities': 12000,
    'food_dining': 7500,
    'transportation': 3000,
    'healthcare': 2000,
    'entertainment': 1500,
    'shopping_discretionary': 1000,
    'credit_card_utilization': 22.0
}

result = service.predict(user_input, model_name="Decision Tree")
print("Predicted Archetype :", result['predicted_category'])
print("Confidence          :", result['confidence'], "%")
print("Financial Health    :", result['health_score'], "/ 100")
```

---

## 📚 In-Depth Project Documentation

For exhaustive, easy-to-understand explanations of every concept, file, and model:
- 📖 [**1. ML Topics Explained (How, Why, When)**](docs/1_ML_TOPICS_EXPLAINED.md)
- 📁 [**2. Project Files Explained (How, Why, When)**](docs/2_PROJECT_FILES_EXPLAINED.md)
- 🤖 [**3. ML Models Explained (How, Why, When)**](docs/3_ML_MODELS_EXPLAINED.md)
- 📑 [**Academic Project Report**](PROJECT_REPORT.md)
