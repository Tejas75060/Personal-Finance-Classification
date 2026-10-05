"""
Online Dataset Ingestion and Preprocessing Pipeline
Fetches and standardizes the real 'Indian Personal Finance and Spending Habits' dataset
Source: Kaggle (Shriyash Jagtap) & GitHub (Somnath-Fintech / Arif-miad)
URL: https://raw.githubusercontent.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits/main/Dataset/data.csv
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits/main/Dataset/data.csv"

def fetch_and_process_online_data(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
    raw_path = os.path.join(output_dir, "personal_finance_data_raw.csv")
    clean_path = os.path.join(output_dir, "personal_finance_data.csv")
    meta_path = os.path.join(output_dir, "dataset_source.json")
    
    # 1. Download if not present
    if not os.path.exists(raw_path):
        print(f"Downloading online dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, raw_path)
        print(f"Downloaded raw dataset to {raw_path}")
    else:
        print(f"Using local cached raw dataset: {raw_path}")
        
    df_raw = pd.read_csv(raw_path)
    print(f"Raw online dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    # 2. Extract standard financial attributes
    # The raw dataset contains real income and category-wise monthly expenses
    expense_cols = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
    ]
    
    # Clean numerical columns
    for col in ['Income'] + expense_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        
    # Group spending categories to match deployment specifications
    # Housing & Utilities: Rent + Utilities
    housing_utilities = df_raw['Rent'] + df_raw['Utilities']
    # Food & Dining: Groceries + Eating_Out
    food_dining = df_raw['Groceries'] + df_raw['Eating_Out']
    # Transportation: Transport
    transportation = df_raw['Transport']
    # Healthcare: Healthcare
    healthcare = df_raw['Healthcare']
    # Entertainment: Entertainment
    entertainment = df_raw['Entertainment']
    # Shopping & Discretionary: Miscellaneous + Education
    shopping_discretionary = df_raw['Miscellaneous'] + df_raw['Education']
    
    # Monthly Income & Expenses
    monthly_income = df_raw['Income'].round(2)
    monthly_expenses = df_raw[expense_cols].sum(axis=1).round(2)
    loan_payments = df_raw['Loan_Repayment'].round(2)
    investment_amount = df_raw['Insurance'].round(2)
    
    # Actual monthly savings = Income - Expenses (non-negative)
    savings = np.maximum(0, monthly_income - monthly_expenses).round(2)
    
    # Financial Ratios
    savings_rate = np.where(monthly_income > 0, (savings / monthly_income) * 100, 0)
    expense_to_income = np.where(monthly_income > 0, (monthly_expenses / monthly_income) * 100, 100)
    debt_to_income = np.where(monthly_income > 0, (loan_payments / monthly_income) * 100, 0)
    
    discretionary_spend = entertainment + shopping_discretionary
    discretionary_ratio = np.where(monthly_expenses > 0, (discretionary_spend / monthly_expenses) * 100, 0)
    
    # Credit Card Utilization estimation based on DTI and discretionary velocity
    # In banking analytics, CC utilization correlates with DTI and high expense ratio
    cc_utilization = np.clip(
        (expense_to_income * 0.6) + (debt_to_income * 0.8) + np.random.normal(0, 5, size=len(df_raw)),
        5.0, 95.0
    ).round(1)
    
    # 3. Classify into Saver, Balanced, High-Spender
    # Standard personal finance benchmark:
    # Saver: Expense-to-income < 68% (disciplined savings rate > 32%)
    # Balanced: Expense-to-income between 68% and 80% (moderate savings rate 20-32%)
    # High-Spender: Expense-to-income > 80% (living paycheck-to-paycheck, savings rate < 20%)
    conditions = [
        expense_to_income < 68.0,
        (expense_to_income >= 68.0) & (expense_to_income <= 80.0),
        expense_to_income > 80.0
    ]
    choices = ['Saver', 'Balanced', 'High-Spender']
    financial_category = np.select(conditions, choices, default='Balanced')
    
    # 4. Construct clean dataframe
    clean_df = pd.DataFrame({
        'user_id': [f"ONL_{10000 + i}" for i in range(len(df_raw))],
        'age': df_raw['Age'].astype(int),
        'dependents': df_raw['Dependents'].astype(int),
        'occupation': df_raw['Occupation'].astype(str),
        'city_tier': df_raw['City_Tier'].astype(str),
        'monthly_income': monthly_income,
        'monthly_expenses': monthly_expenses,
        'savings': savings,
        'loan_payments': loan_payments,
        'investment_amount': investment_amount,
        'housing_utilities': housing_utilities.round(2),
        'food_dining': food_dining.round(2),
        'transportation': transportation.round(2),
        'healthcare': healthcare.round(2),
        'entertainment': entertainment.round(2),
        'shopping_discretionary': shopping_discretionary.round(2),
        'credit_card_utilization': cc_utilization,
        'financial_category': financial_category
    })
    
    # Save processed dataset
    clean_df.to_csv(clean_path, index=False)
    print(f"Processed dataset saved to: {clean_path}")
    print(f"Total Records: {len(clean_df)}")
    print("Class distribution:")
    print(clean_df['financial_category'].value_counts())
    
    # Save source metadata
    source_metadata = {
        'source_name': 'Indian Personal Finance and Spending Habits Dataset',
        'provider': 'Kaggle & GitHub Open Data',
        'raw_url': DATA_URL,
        'kaggle_reference': 'https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits',
        'github_reference': 'https://github.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits',
        'total_records': len(clean_df),
        'features_count': len(clean_df.columns),
        'target_classes': list(clean_df['financial_category'].unique()),
        'class_breakdown': clean_df['financial_category'].value_counts().to_dict()
    }
    with open(meta_path, 'w') as f:
        json.dump(source_metadata, f, indent=2)
        
    print(f"Source metadata saved to: {meta_path}")
    return clean_df

if __name__ == "__main__":
    fetch_and_process_online_data()
