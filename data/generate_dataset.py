"""
Synthetic Financial Dataset Generator for Personal Finance Classification
Generates realistic financial records for machine learning classification.
Target Classes: Saver, Balanced, High-Spender
"""

import numpy as np
import pandas as pd
import random
import os

def generate_financial_data(n_samples=3500, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    
    records = []
    
    # Target distribution: ~33% Saver, ~37% Balanced, ~30% High-Spender
    categories = ['Saver', 'Balanced', 'High-Spender']
    class_probs = [0.33, 0.37, 0.30]
    
    assigned_classes = np.random.choice(categories, size=n_samples, p=class_probs)
    
    for i, target_class in enumerate(assigned_classes):
        # Monthly Income: Log-normal distribution to reflect real income distributions
        # Median ~ $5,500, range ~$2,000 to $22,000
        base_income = float(np.random.lognormal(mean=8.6, sigma=0.45))
        monthly_income = round(np.clip(base_income, 2200, 25000), 2)
        
        # User age and financial profile
        age = int(np.clip(np.random.normal(35, 10), 20, 68))
        dependents = int(np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.25, 0.25, 0.10, 0.05]))
        
        if target_class == 'Saver':
            # High savings & investment, low discretionary spending
            savings_ratio = np.random.uniform(0.25, 0.48)
            investment_ratio = np.random.uniform(0.12, 0.30)
            debt_ratio = np.random.uniform(0.02, 0.15)
            
            # Expenses breakdown (% of total expenses)
            # Savers prioritize essentials, low entertainment & shopping
            exp_to_inc_ratio = max(0.25, 1.0 - (savings_ratio + investment_ratio) + np.random.normal(0, 0.03))
            exp_to_inc_ratio = min(0.60, exp_to_inc_ratio)
            monthly_expenses = round(monthly_income * exp_to_inc_ratio, 2)
            
            housing_pct = np.random.uniform(0.40, 0.55)
            food_pct = np.random.uniform(0.20, 0.30)
            transport_pct = np.random.uniform(0.08, 0.14)
            healthcare_pct = np.random.uniform(0.06, 0.12)
            entertainment_pct = np.random.uniform(0.03, 0.08)
            shopping_pct = np.random.uniform(0.03, 0.09)
            
            credit_card_utilization = round(np.random.uniform(4.0, 24.0), 1)
            
        elif target_class == 'Balanced':
            # Moderate savings & investment, healthy balance
            savings_ratio = np.random.uniform(0.14, 0.25)
            investment_ratio = np.random.uniform(0.07, 0.16)
            debt_ratio = np.random.uniform(0.10, 0.28)
            
            exp_to_inc_ratio = max(0.55, 1.0 - (savings_ratio + investment_ratio) + np.random.normal(0, 0.04))
            exp_to_inc_ratio = min(0.78, exp_to_inc_ratio)
            monthly_expenses = round(monthly_income * exp_to_inc_ratio, 2)
            
            housing_pct = np.random.uniform(0.35, 0.48)
            food_pct = np.random.uniform(0.18, 0.26)
            transport_pct = np.random.uniform(0.10, 0.18)
            healthcare_pct = np.random.uniform(0.05, 0.11)
            entertainment_pct = np.random.uniform(0.08, 0.16)
            shopping_pct = np.random.uniform(0.08, 0.18)
            
            credit_card_utilization = round(np.random.uniform(20.0, 48.0), 1)
            
        else: # High-Spender
            # High expenses, low savings, high discretionary spending
            savings_ratio = np.random.uniform(0.01, 0.09)
            investment_ratio = np.random.uniform(0.00, 0.06)
            debt_ratio = np.random.uniform(0.18, 0.42)
            
            exp_to_inc_ratio = max(0.78, 1.0 - (savings_ratio + investment_ratio) + np.random.normal(0.05, 0.04))
            exp_to_inc_ratio = min(1.15, exp_to_inc_ratio)  # Some live beyond means
            monthly_expenses = round(monthly_income * exp_to_inc_ratio, 2)
            
            housing_pct = np.random.uniform(0.28, 0.40)
            food_pct = np.random.uniform(0.16, 0.24)
            transport_pct = np.random.uniform(0.10, 0.20)
            healthcare_pct = np.random.uniform(0.04, 0.09)
            entertainment_pct = np.random.uniform(0.16, 0.32)
            shopping_pct = np.random.uniform(0.18, 0.35)
            
            credit_card_utilization = round(np.random.uniform(50.0, 96.0), 1)
            
        # Normalize category percentages to sum exactly to 1.0
        pct_sum = housing_pct + food_pct + transport_pct + healthcare_pct + entertainment_pct + shopping_pct
        housing_pct /= pct_sum
        food_pct /= pct_sum
        transport_pct /= pct_sum
        healthcare_pct /= pct_sum
        entertainment_pct /= pct_sum
        shopping_pct /= pct_sum
        
        # Calculate dollar amounts for categories
        housing_utilities = round(monthly_expenses * housing_pct, 2)
        food_dining = round(monthly_expenses * food_pct, 2)
        transportation = round(monthly_expenses * transport_pct, 2)
        healthcare = round(monthly_expenses * healthcare_pct, 2)
        entertainment = round(monthly_expenses * entertainment_pct, 2)
        shopping_discretionary = round(monthly_expenses * shopping_pct, 2)
        
        # Calculate savings, debt payments, and investments
        savings = round(monthly_income * savings_ratio, 2)
        loan_payments = round(monthly_income * debt_ratio, 2)
        investment_amount = round(monthly_income * investment_ratio, 2)
        
        # Add realistic boundary noise (occasional borderline transitions)
        # ~3.5% label noise to simulate realistic messy real-world survey/banking data
        final_class = target_class
        if np.random.rand() < 0.035:
            if target_class == 'Saver':
                final_class = 'Balanced'
            elif target_class == 'High-Spender':
                final_class = 'Balanced'
            else:
                final_class = np.random.choice(['Saver', 'High-Spender'])

        records.append({
            'user_id': f"USR_{10000 + i}",
            'age': age,
            'dependents': dependents,
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'savings': savings,
            'loan_payments': loan_payments,
            'investment_amount': investment_amount,
            'housing_utilities': housing_utilities,
            'food_dining': food_dining,
            'transportation': transportation,
            'healthcare': healthcare,
            'entertainment': entertainment,
            'shopping_discretionary': shopping_discretionary,
            'credit_card_utilization': credit_card_utilization,
            'financial_category': final_class
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_financial_data(n_samples=3600, random_state=42)
    output_path = os.path.join(os.path.dirname(__file__), "personal_finance_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records.")
    print("Class distribution:")
    print(df['financial_category'].value_counts())
    print("\nSample records:")
    print(df.head(3))
