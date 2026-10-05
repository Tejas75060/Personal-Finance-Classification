"""
Prediction & Financial Advisory Service
Loads pre-trained models and executes inference, financial health scoring,
and tailored advisory recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class FinancialClassifierService:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            
        self.models_dir = models_dir
        with open(os.path.join(models_dir, "metrics.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.feature_cols = self.metadata['feature_cols']
        self.classes = self.metadata['classes']
        self.best_model_name = self.metadata['best_model']
        
        self.models = {}
        need_retrain = False
        
        # Resilient model loading with fallback for cross-version Python/scikit-learn environments
        try:
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
            self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
            
            for name in self.metadata['results'].keys():
                slug = name.lower().replace(' ', '_').replace('-', '_')
                model_path = os.path.join(models_dir, f"{slug}_model.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    
            if len(self.models) < len(self.metadata['results']):
                need_retrain = True
        except Exception as e:
            print(f"Pickle compatibility mismatch detected ({e}). Training models natively in current environment...")
            need_retrain = True
            
        if need_retrain:
            self._train_in_environment()

    def _train_in_environment(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "personal_finance_data.csv")
        if not os.path.exists(csv_path):
            return
            
        df = pd.read_csv(csv_path)
        if len(df) > 8000:
            df = df.sample(n=8000, random_state=42)
            
        df['savings_rate'] = (df['savings'] / df['monthly_income']) * 100
        df['expense_to_income'] = (df['monthly_expenses'] / df['monthly_income']) * 100
        df['debt_to_income'] = (df['loan_payments'] / df['monthly_income']) * 100
        df['investment_rate'] = (df['investment_amount'] / df['monthly_income']) * 100
        
        discretionary_spend = df['entertainment'] + df['shopping_discretionary']
        df['discretionary_ratio'] = (discretionary_spend / df['monthly_expenses']) * 100
        
        essential_spend = df['housing_utilities'] + df['food_dining'] + df['healthcare'] + df['transportation']
        df['essential_ratio'] = (essential_spend / df['monthly_expenses']) * 100

        X = df[self.feature_cols]
        le = LabelEncoder()
        y = le.fit_transform(df['financial_category'])
        self.label_encoder = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        configs = {
            'Decision Tree': (DecisionTreeClassifier(max_depth=6, random_state=42), False),
            'Random Forest': (RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42), False),
            'Gradient Boosting': (GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42), False),
            'Logistic Regression': (LogisticRegression(max_iter=500, random_state=42), True),
            'K-Nearest Neighbors': (KNeighborsClassifier(n_neighbors=7, weights='distance'), True),
            'Support Vector Machine': (SVC(kernel='rbf', C=1.5, probability=True, random_state=42), True)
        }

        self.models = {}
        for name, (clf, use_scaled) in configs.items():
            X_tr = X_scaled if use_scaled else X
            if name == 'Support Vector Machine' and len(X_tr) > 2500:
                idx = np.random.RandomState(42).choice(len(X_tr), size=2500, replace=False)
                clf.fit(X_tr[idx], y[idx])
            else:
                clf.fit(X_tr, y)
            self.models[name] = clf
            
        print("Models successfully trained and ready in current environment.")

    def extract_features(self, input_data):
        income = float(input_data.get('monthly_income', 5000))
        expenses = float(input_data.get('monthly_expenses', 3000))
        savings = float(input_data.get('savings', 1000))
        loans = float(input_data.get('loan_payments', 500))
        investments = float(input_data.get('investment_amount', 500))
        
        # Categories
        housing = float(input_data.get('housing_utilities', expenses * 0.40))
        food = float(input_data.get('food_dining', expenses * 0.22))
        trans = float(input_data.get('transportation', expenses * 0.12))
        health = float(input_data.get('healthcare', expenses * 0.08))
        ent = float(input_data.get('entertainment', expenses * 0.09))
        shop = float(input_data.get('shopping_discretionary', expenses * 0.09))
        
        cc_util = float(input_data.get('credit_card_utilization', 30.0))
        
        # Guard against zero income division
        safe_income = max(income, 1.0)
        safe_expenses = max(expenses, 1.0)
        
        # Engineered ratios
        savings_rate = (savings / safe_income) * 100
        expense_to_income = (expenses / safe_income) * 100
        debt_to_income = (loans / safe_income) * 100
        investment_rate = (investments / safe_income) * 100
        
        discretionary_spend = ent + shop
        discretionary_ratio = (discretionary_spend / safe_expenses) * 100
        
        essential_spend = housing + food + trans + health
        essential_ratio = (essential_spend / safe_expenses) * 100
        
        feature_dict = {
            'monthly_income': income,
            'monthly_expenses': expenses,
            'savings': savings,
            'loan_payments': loans,
            'investment_amount': investments,
            'housing_utilities': housing,
            'food_dining': food,
            'transportation': trans,
            'healthcare': health,
            'entertainment': ent,
            'shopping_discretionary': shop,
            'credit_card_utilization': cc_util,
            'savings_rate': savings_rate,
            'expense_to_income': expense_to_income,
            'debt_to_income': debt_to_income,
            'investment_rate': investment_rate,
            'discretionary_ratio': discretionary_ratio,
            'essential_ratio': essential_ratio
        }
        
        # Return dataframe ordered matching feature_cols
        X_df = pd.DataFrame([feature_dict])[self.feature_cols]
        return X_df, feature_dict

    def calculate_health_score(self, metrics):
        score = 0
        
        # 1. Savings Rate (25 pts): Ideal >= 20%
        sr = metrics['savings_rate']
        if sr >= 25:
            score += 25
        elif sr >= 15:
            score += 18
        elif sr >= 8:
            score += 10
        elif sr > 0:
            score += 5
            
        # 2. Expense to Income (25 pts): Ideal <= 60%
        ei = metrics['expense_to_income']
        if ei <= 50:
            score += 25
        elif ei <= 65:
            score += 20
        elif ei <= 80:
            score += 14
        elif ei <= 95:
            score += 6
        else:
            score += 0
            
        # 3. Debt to Income (20 pts): Ideal <= 20%
        dti = metrics['debt_to_income']
        if dti <= 15:
            score += 20
        elif dti <= 28:
            score += 15
        elif dti <= 36:
            score += 9
        elif dti <= 45:
            score += 4
        else:
            score += 0
            
        # 4. Investment Rate (15 pts): Ideal >= 10%
        ir = metrics['investment_rate']
        if ir >= 15:
            score += 15
        elif ir >= 8:
            score += 10
        elif ir >= 3:
            score += 5
        else:
            score += 0
            
        # 5. Discretionary Control (15 pts): Ideal <= 25%
        disc = metrics['discretionary_ratio']
        if disc <= 20:
            score += 15
        elif disc <= 30:
            score += 11
        elif disc <= 40:
            score += 6
        else:
            score += 2
            
        return int(np.clip(score, 0, 100))

    def generate_recommendations(self, category, metrics):
        recs = []
        alerts = []
        
        sr = metrics['savings_rate']
        dti = metrics['debt_to_income']
        ei = metrics['expense_to_income']
        cc = metrics['credit_card_utilization']
        disc = metrics['discretionary_ratio']
        
        # Risk Alerts
        if ei > 90:
            alerts.append("Critical Expense Ratio: You are spending over 90% of your earnings, leaving negligible safety margin.")
        if dti > 36:
            alerts.append("Elevated Debt-to-Income: Your monthly loan obligations exceed 36% of gross income, posing high financial vulnerability.")
        if cc > 40:
            alerts.append(f"High Credit Card Utilization ({cc}%): Consistently exceeding 30% utilization negatively impacts credit score and incurs heavy APR interest.")
            
        if category == 'High-Spender':
            recs.append({
                'title': 'Implement Strict 50/30/20 Budget Restructuring',
                'description': 'Target capping fixed needs at 50%, discretionary desires at 30%, and channel the remaining 20% toward debt reduction and liquid reserves.',
                'tag': 'Budgeting'
            })
            recs.append({
                'title': 'Audit Discretionary Spending',
                'description': f'Your discretionary spending stands at {disc:.1f}% of total expenses. Audit recurring subscriptions, fine dining, and impulse online retail to free up cash flow.',
                'tag': 'Spending'
            })
            recs.append({
                'title': 'Prioritize High-Interest Debt Avalanche',
                'description': 'Allocate excess monthly funds to wipe out credit cards and personal loans charging high interest rates before embarking on speculative investments.',
                'tag': 'Debt Management'
            })
            recs.append({
                'title': 'Establish a 3-Month Emergency Cushion',
                'description': 'Automate a weekly deposit of at least 10% of income into a dedicated high-yield savings account until you reach 3 months of basic living expenses.',
                'tag': 'Savings'
            })
            
        elif category == 'Balanced':
            recs.append({
                'title': 'Optimize Wealth Accumulation via Index Investing',
                'description': f'With a solid savings rate of {sr:.1f}%, step up monthly contributions to broad-market index funds (e.g. S&P 500, Total Market) or tax-advantaged retirement accounts.',
                'tag': 'Investing'
            })
            recs.append({
                'title': 'Maintain Shield Against Lifestyle Inflation',
                'description': 'Whenever income rises, commit at least 50% of the raise to savings and investing before expanding discretionary lifestyle choices.',
                'tag': 'Growth'
            })
            recs.append({
                'title': 'Fine-Tune Debt Amortization',
                'description': f'Your DTI of {dti:.1f}% is manageable. Look for opportunities to refinance higher-rate loans or make lump-sum principal paydowns to eliminate loan tenure faster.',
                'tag': 'Debt'
            })
            recs.append({
                'title': 'Goal-Based Bucket Strategy',
                'description': 'Separate your capital into three distinct buckets: Short-term cash (1-2 yrs), Medium-term bonds/fixed deposits (3-5 yrs), and Long-term equities (7+ yrs).',
                'tag': 'Strategy'
            })
            
        else: # Saver
            recs.append({
                'title': 'Combat Cash Drag with Asset Allocation',
                'description': f'Your savings rate of {sr:.1f}% is commendable. Ensure surplus liquidity beyond 6 months of expenses is invested in inflation-beating growth assets rather than stagnant cash.',
                'tag': 'Wealth Growth'
            })
            recs.append({
                'title': 'Maximize Tax-Advantaged Investment Vehicles',
                'description': 'Fully utilize available annual limits for tax-sheltered accounts (401(k), IRA, HSA, or PPF/ELSS depending on jurisdiction) for maximum compound tax efficiency.',
                'tag': 'Tax Optimization'
            })
            recs.append({
                'title': 'Healthy Balance: Invest in Experiences & Skills',
                'description': 'While financial discipline is exemplary, consider budgeting intentional funds for career upskilling, personal wellness, and high-value life experiences.',
                'tag': 'Lifestyle'
            })
            recs.append({
                'title': 'Long-Term Estate & Insurance Planning',
                'description': 'Ensure term life insurance and comprehensive health coverage adequately protect your growing net worth against unforeseen liability or medical shocks.',
                'tag': 'Protection'
            })
            
        return recs, alerts

    def predict(self, input_data, model_name=None):
        if model_name is None or model_name not in self.models:
            model_name = self.best_model_name
            
        model = self.models[model_name]
        model_info = self.metadata['results'][model_name]
        use_scaled = model_info['use_scaled']
        
        X_df, metrics = self.extract_features(input_data)
        
        if use_scaled:
            X_input = self.scaler.transform(X_df)
        else:
            X_input = X_df
            
        pred_encoded = model.predict(X_input)[0]
        predicted_category = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Probabilities
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            for cls_name, prob in zip(self.label_encoder.classes_, probs):
                probabilities[cls_name] = round(float(prob) * 100, 1)
        else:
            for cls_name in self.label_encoder.classes_:
                probabilities[cls_name] = 100.0 if cls_name == predicted_category else 0.0
                
        health_score = self.calculate_health_score(metrics)
        recommendations, alerts = self.generate_recommendations(predicted_category, metrics)
        
        # Benchmark 50/30/20 breakdown
        ideal_needs = round(metrics['monthly_income'] * 0.50, 2)
        ideal_wants = round(metrics['monthly_income'] * 0.30, 2)
        ideal_savings = round(metrics['monthly_income'] * 0.20, 2)
        
        actual_needs = round(metrics['housing_utilities'] + metrics['food_dining'] + metrics['transportation'] + metrics['healthcare'] + metrics['loan_payments'], 2)
        actual_wants = round(metrics['entertainment'] + metrics['shopping_discretionary'], 2)
        actual_savings = round(metrics['savings'] + metrics['investment_amount'], 2)
        
        return {
            'predicted_category': predicted_category,
            'confidence': probabilities.get(predicted_category, 100.0),
            'model_used': model_name,
            'probabilities': probabilities,
            'health_score': health_score,
            'metrics': {k: round(v, 2) for k, v in metrics.items()},
            'recommendations': recommendations,
            'alerts': alerts,
            'budget_comparison': {
                'benchmark': {'needs': ideal_needs, 'wants': ideal_wants, 'savings_investments': ideal_savings},
                'actual': {'needs': actual_needs, 'wants': actual_wants, 'savings_investments': actual_savings}
            }
        }

if __name__ == "__main__":
    service = FinancialClassifierService()
    print("Service initialized.")
    sample = {
        'monthly_income': 6500,
        'monthly_expenses': 2800,
        'savings': 2200,
        'loan_payments': 400,
        'investment_amount': 1100,
        'housing_utilities': 1200,
        'food_dining': 600,
        'transportation': 350,
        'healthcare': 250,
        'entertainment': 200,
        'shopping_discretionary': 200,
        'credit_card_utilization': 15.0
    }
    res = service.predict(sample)
    print("\nSample Prediction Result:")
    print(f"Predicted Category: {res['predicted_category']} (Confidence: {res['confidence']}%)")
    print(f"Health Score: {res['health_score']}/100")
    print(f"Probabilities: {res['probabilities']}")
