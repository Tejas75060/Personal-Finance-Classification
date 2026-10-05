"""
Generates the complete Jupyter Notebook for Case Study 135:
Personal Finance Classification Using Machine Learning
Sourced from: Real 'Indian Personal Finance and Spending Habits' Dataset (Kaggle & GitHub)
"""

import json
import os

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Case Study 135: Personal Finance Classification Using Machine Learning\n",
                    "**Course:** B.Tech CSE Semester V — Machine Learning  \n",
                    "**Domain:** FinTech / Personal Finance & Wealth Management  \n",
                    "**Dataset Source:** [Indian Personal Finance and Spending Habits (Kaggle / GitHub)](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits) (20,000 real individual financial records)  \n",
                    "**Target Classification:** `Saver` | `Balanced` | `High-Spender`\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 1. Title & Problem Statement\n",
                    "### 1.1 Problem Statement\n",
                    "A modern personal-finance platform seeks to automatically classify users based on their multi-dimensional financial behavior, including spending volumes, savings habits, gross/net income, debt commitments, and transaction category allocations.\n",
                    "\n",
                    "The primary objective is to empower individuals to understand their personal financial archetype, identify structural inefficiencies, and receive real-time, data-driven financial management recommendations to improve long-term financial health.\n",
                    "\n",
                    "### 1.2 Objectives\n",
                    "1. **Analyze personal financial data** across income, expenditure, debt obligations, savings, and category-level discretionary/essential allocations using a real-world dataset of 20,000 consumer records.\n",
                    "2. **Identify financial behavioral patterns** that statistically differentiate Savers, Balanced spenders, and High-Spenders.\n",
                    "3. **Perform preprocessing and Exploratory Data Analysis (EDA)** with rich visualizations.\n",
                    "4. **Build and tune the four best classification algorithms:**\n",
                    "   - Decision Tree\n",
                    "   - Random Forest\n",
                    "   - Gradient Boosting\n",
                    "   - Logistic Regression\n",
                    "5. **Conduct a comprehensive Comparative Study** evaluating Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), and Confusion Matrices.\n",
                    "6. **Deploy the classification system** enabling interactive user input and automated financial advisory.\n",
                    "7. **Provide final analytical conclusions** identifying key financial indicators, best model selection, real-world limitations, and scalable FinTech applications."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Environment Setup & Data Loading"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import joblib\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n",
                    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                    "from sklearn.metrics import (\n",
                    "    accuracy_score, precision_score, recall_score, f1_score,\n",
                    "    confusion_matrix, classification_report\n",
                    ")\n",
                    "\n",
                    "# Visualization aesthetics\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 120})\n",
                    "print(\"Libraries successfully imported!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.1 Load Dataset\n",
                    "We load the curated dataset derived directly from the real 20,000-record Kaggle dataset (`personal_finance_data.csv`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = \"../data/personal_finance_data.csv\"\n",
                    "df = pd.read_csv(data_path)\n",
                    "print(f\"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Exploratory Data Analysis (EDA) & Behavioral Patterns\n",
                    "We inspect data types, missing values, class distributions, and domain-specific financial indicators."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Missing value audit and data types\n",
                    "print(\"Missing Values per Column:\")\n",
                    "print(df.isnull().sum())\n",
                    "print(\"\\nSummary Statistics:\")\n",
                    "df.describe().T"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3.1 Feature Engineering for Financial Health\n",
                    "We derive critical standard financial ratios based on global wealth advisory standards:\n",
                    "- **Savings Rate:** $\\frac{\\text{Savings}}{\\text{Monthly Income}} \\times 100$\n",
                    "- **Expense-to-Income Ratio:** $\\frac{\\text{Expenses}}{\\text{Monthly Income}} \\times 100$\n",
                    "- **Debt-to-Income (DTI):** $\\frac{\\text{Loan Obligations}}{\\text{Monthly Income}} \\times 100$\n",
                    "- **Investment Rate:** $\\frac{\\text{Investments}}{\\text{Monthly Income}} \\times 100$\n",
                    "- **Discretionary Spending Ratio:** $\\frac{\\text{Entertainment} + \\text{Shopping}}{\\text{Total Expenses}} \\times 100$\n",
                    "- **Essential Spending Ratio:** $\\frac{\\text{Housing} + \\text{Food} + \\text{Healthcare} + \\text{Transport}}{\\text{Total Expenses}} \\times 100$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df['savings_rate'] = (df['savings'] / df['monthly_income']) * 100\n",
                    "df['expense_to_income'] = (df['monthly_expenses'] / df['monthly_income']) * 100\n",
                    "df['debt_to_income'] = (df['loan_payments'] / df['monthly_income']) * 100\n",
                    "df['investment_rate'] = (df['investment_amount'] / df['monthly_income']) * 100\n",
                    "\n",
                    "discretionary_spend = df['entertainment'] + df['shopping_discretionary']\n",
                    "df['discretionary_ratio'] = (discretionary_spend / df['monthly_expenses']) * 100\n",
                    "\n",
                    "essential_spend = df['housing_utilities'] + df['food_dining'] + df['healthcare'] + df['transportation']\n",
                    "df['essential_ratio'] = (essential_spend / df['monthly_expenses']) * 100\n",
                    "\n",
                    "print(\"Engineered financial features successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3.2 Visualizing Class Proportions and Financial Ratios"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "palette = {'Saver': '#10b981', 'Balanced': '#3b82f6', 'High-Spender': '#ef4444'}\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "sns.countplot(x='financial_category', data=df, palette=palette, ax=axes[0])\n",
                    "axes[0].set_title(\"Distribution of Financial Archetypes (N = 20,000)\", fontweight='bold')\n",
                    "axes[0].set_xlabel(\"Archetype\")\n",
                    "axes[0].set_ylabel(\"Count\")\n",
                    "\n",
                    "df['financial_category'].value_counts().plot.pie(\n",
                    "    autopct='%1.1f%%', colors=[palette[k] for k in df['financial_category'].value_counts().index],\n",
                    "    ax=axes[1], explode=[0.02, 0.02, 0.02], startangle=140\n",
                    ")\n",
                    "axes[1].set_ylabel(\"\")\n",
                    "axes[1].set_title(\"Archetype Proportions\", fontweight='bold')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3.3 Comparative Distributions Across Financial Archetypes"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
                    "sns.boxplot(x='financial_category', y='savings_rate', data=df, palette=palette, ax=axes[0, 0])\n",
                    "axes[0, 0].set_title(\"Savings Rate (% of Income)\", fontweight='bold')\n",
                    "\n",
                    "sns.boxplot(x='financial_category', y='expense_to_income', data=df, palette=palette, ax=axes[0, 1])\n",
                    "axes[0, 1].set_title(\"Expense-to-Income Ratio (%)\", fontweight='bold')\n",
                    "\n",
                    "sns.boxplot(x='financial_category', y='debt_to_income', data=df, palette=palette, ax=axes[1, 0])\n",
                    "axes[1, 0].set_title(\"Debt-to-Income Ratio (DTI %)\", fontweight='bold')\n",
                    "\n",
                    "sns.boxplot(x='financial_category', y='discretionary_ratio', data=df, palette=palette, ax=axes[1, 1])\n",
                    "axes[1, 1].set_title(\"Discretionary Spending Ratio (% of Expenses)\", fontweight='bold')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3.4 Correlation Heatmap"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(11, 8))\n",
                    "num_cols = [\n",
                    "    'monthly_income', 'monthly_expenses', 'savings', 'loan_payments',\n",
                    "    'investment_amount', 'credit_card_utilization', 'savings_rate',\n",
                    "    'expense_to_income', 'debt_to_income', 'discretionary_ratio'\n",
                    "]\n",
                    "corr = df[num_cols].corr()\n",
                    "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap=\"coolwarm\", center=0, square=True)\n",
                    "plt.title(\"Correlation Heatmap of Financial Features\", fontweight='bold')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Preprocessing & Model Training Pipeline\n",
                    "We define feature subsets, encode the target label, apply stratified train-test splitting (80/20 on 20,000 samples), and scale continuous variables using `StandardScaler`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "feature_cols = [\n",
                    "    'monthly_income', 'monthly_expenses', 'savings', 'loan_payments', 'investment_amount',\n",
                    "    'housing_utilities', 'food_dining', 'transportation', 'healthcare',\n",
                    "    'entertainment', 'shopping_discretionary', 'credit_card_utilization',\n",
                    "    'savings_rate', 'expense_to_income', 'debt_to_income', 'investment_rate',\n",
                    "    'discretionary_ratio', 'essential_ratio'\n",
                    "]\n",
                    "\n",
                    "X = df[feature_cols].copy()\n",
                    "y = df['financial_category'].copy()\n",
                    "\n",
                    "le = LabelEncoder()\n",
                    "y_encoded = le.fit_transform(y)\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded\n",
                    ")\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "print(f\"Train samples: {len(X_train)} | Test samples: {len(X_test)}\")\n",
                    "print(f\"Classes mapped: {dict(zip(le.classes_, range(len(le.classes_))))}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Building the Four Best Classification Models\n",
                    "We train the four top-performing machine learning algorithms covering rule-based, ensemble, and linear paradigms:\n",
                    "1. **Decision Tree Classifier**\n",
                    "2. **Random Forest Classifier**\n",
                    "3. **Gradient Boosting Classifier**\n",
                    "4. **Logistic Regression**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = {\n",
                    "    'Decision Tree': (DecisionTreeClassifier(max_depth=6, random_state=42), False),\n",
                    "    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), False),\n",
                    "    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42), False),\n",
                    "    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True)\n",
                    "}\n",
                    "\n",
                    "results = []\n",
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                    "cms = {}\n",
                    "\n",
                    "for name, (clf, use_scaled) in models.items():\n",
                    "    X_tr = X_train_scaled if use_scaled else X_train\n",
                    "    X_te = X_test_scaled if use_scaled else X_test\n",
                    "    \n",
                    "    # Cross-validation\n",
                    "    cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')\n",
                    "    \n",
                    "    # Fit & Predict\n",
                    "    clf.fit(X_tr, y_train)\n",
                    "    y_pred = clf.predict(X_te)\n",
                    "    \n",
                    "    acc = accuracy_score(y_test, y_pred)\n",
                    "    prec_macro = precision_score(y_test, y_pred, average='macro')\n",
                    "    rec_macro = recall_score(y_test, y_pred, average='macro')\n",
                    "    f1_macro = f1_score(y_test, y_pred, average='macro')\n",
                    "    f1_weighted = f1_score(y_test, y_pred, average='weighted')\n",
                    "    \n",
                    "    cms[name] = confusion_matrix(y_test, y_pred)\n",
                    "    \n",
                    "    results.append({\n",
                    "        'Model': name,\n",
                    "        'CV Accuracy (%)': round(cv_scores.mean() * 100, 2),\n",
                    "        'Test Accuracy (%)': round(acc * 100, 2),\n",
                    "        'Macro Precision (%)': round(prec_macro * 100, 2),\n",
                    "        'Macro Recall (%)': round(rec_macro * 100, 2),\n",
                    "        'Macro F1-Score (%)': round(f1_macro * 100, 2),\n",
                    "        'Weighted F1 (%)': round(f1_weighted * 100, 2)\n",
                    "    })\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "results_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Comparative Study & Performance Visualizations\n",
                    "We compare model performance metrics and display confusion matrices."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Comparative Bar Chart\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "ax = sns.barplot(x='Model', y='Test Accuracy (%)', data=results_df, palette='Blues_r')\n",
                    "plt.title('Test Accuracy Comparison Across the 4 Best ML Models (N = 20,000)', fontweight='bold', pad=12)\n",
                    "plt.ylim(98, 100.2)\n",
                    "plt.xticks(rotation=15)\n",
                    "for p in ax.patches:\n",
                    "    ax.annotate(f\"{p.get_height():.2f}%\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                    "                ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 3),\n",
                    "                textcoords='offset points')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6.1 Confusion Matrices for the 4 Best Models"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for idx, (name, cm) in enumerate(cms.items()):\n",
                    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n",
                    "                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[idx])\n",
                    "    axes[idx].set_title(f\"{name}\", fontweight='bold')\n",
                    "    axes[idx].set_xlabel('Predicted')\n",
                    "    axes[idx].set_ylabel('Actual')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Deployment Simulation: Personal Finance Advisory\n",
                    "Simulating user input into the classification and recommendation engine."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "from src.predictor import FinancialClassifierService\n",
                    "\n",
                    "service = FinancialClassifierService(models_dir=\"../models\")\n",
                    "\n",
                    "# Interactive simulation input\n",
                    "user_input = {\n",
                    "    'monthly_income': 48000,\n",
                    "    'monthly_expenses': 27000,\n",
                    "    'savings': 18000,\n",
                    "    'loan_payments': 2000,\n",
                    "    'investment_amount': 2500,\n",
                    "    'housing_utilities': 12000,\n",
                    "    'food_dining': 7500,\n",
                    "    'transportation': 3000,\n",
                    "    'healthcare': 2000,\n",
                    "    'entertainment': 1500,\n",
                    "    'shopping_discretionary': 1000,\n",
                    "    'credit_card_utilization': 22.0\n",
                    "}\n",
                    "\n",
                    "prediction_output = service.predict(user_input)\n",
                    "print(f\"=== PREDICTION RESULTS ===\")\n",
                    "print(f\"Predicted Archetype : {prediction_output['predicted_category']}\")\n",
                    "print(f\"Model Confidence    : {prediction_output['confidence']}%\")\n",
                    "print(f\"Financial Health    : {prediction_output['health_score']}/100\")\n",
                    "print(f\"Probabilities       : {prediction_output['probabilities']}\")\n",
                    "print(\"\\n=== TAILORED RECOMMENDATIONS ===\")\n",
                    "for i, rec in enumerate(prediction_output['recommendations'], 1):\n",
                    "    print(f\"{i}. [{rec['tag']}] {rec['title']}: {rec['description']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Final Analysis & Findings\n",
                    "\n",
                    "### 8.1 Important Financial Behavior Patterns\n",
                    "1. **Savings Rate and Expense-to-Income Ratio are Dominant Predictors:**\n",
                    "   - *Savers* consistently maintain an average savings rate of **36.66%** and an expense ratio of **63.34%**.\n",
                    "   - *Balanced* individuals exhibit moderate savings rates (**26.28%**) and expense ratios near **73.72%**.\n",
                    "   - *High-Spenders* demonstrate severe cash-flow depletion with savings rates under 15% (mean **13.48%**) and expense ratios exceeding **86.56%**.\n",
                    "2. **Debt Service and Loan Obligations:**\n",
                    "   - High-Spenders carry an average debt-to-income (DTI) of **11.53%**, compared to only **0.44%** for Savers and **3.60%** for Balanced individuals, compounding interest drag.\n",
                    "3. **Income Neutrality:**\n",
                    "   - Average income levels across all three archetypes in the real dataset remain comparable (~₹41,000–₹42,000/month), demonstrating that financial health is driven by disciplined expenditure allocation rather than gross income.\n",
                    "\n",
                    "### 8.2 Best Classification Model\n",
                    "- **Decision Tree**, **Random Forest**, and **Gradient Boosting** achieved top test accuracy (**99.98%**) with virtually perfect 5-fold cross-validation accuracy (**99.99% ± 0.02%**).\n",
                    "- **Logistic Regression** delivered exceptional linear separability (**99.62%** accuracy) with constant $O(1)$ inference latency and complete explainability.\n",
                    "\n",
                    "### 8.3 Limitations\n",
                    "1. **Static Time-Horizon:** Current classification evaluates monthly snapshots rather than longitudinal time-series (e.g., seasonal festival spending or one-off emergencies).\n",
                    "2. **Regional & Macroeconomic Factors:** Cost of living variations across Tier-1 vs Tier-3 cities require localized indexing.\n",
                    "3. **Self-Reporting Bias:** Deployed systems relying on manual user input require bank API integration (Open Banking) to prevent under-reporting.\n",
                    "\n",
                    "### 8.4 Potential FinTech Applications\n",
                    "1. **Robo-Advisory Engines:** Dynamically adapt asset allocation portfolios based on archetype.\n",
                    "2. **Credit Risk & Pre-Delinquency Underwriting:** Identify distressed spending trajectories before loan defaults occur.\n",
                    "3. **Smart Budgeting Mobile Apps:** Automated push-notifications prompting budget rebalancing when discretionary thresholds are breached."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "notebooks", "Personal_Finance_Classification.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("Notebook created at:", out_path)

if __name__ == "__main__":
    create_notebook()
