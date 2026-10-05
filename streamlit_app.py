"""
Personal Finance Classification System - Streamlit Application
Interactive FinTech Behavioral Classifier and Advisory Studio.
Course: B.Tech CSE Semester V - Machine Learning (Case Study 135)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

from src.predictor import FinancialClassifierService

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Finance Classification | ML System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism & dark aesthetics
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #0a0e17;
    }
    
    /* Custom Headers */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Badges */
    .badge-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.2rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Result Cards */
    .archetype-card-saver {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .archetype-card-balanced {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .archetype-card-spender {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .metric-container {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Alert cards */
    .alert-box {
        background: rgba(244, 63, 94, 0.15);
        border-left: 4px solid #f43f5e;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #fecdd3;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }
    
    /* Recommendation Card */
    .rec-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Service & Metadata
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

@st.cache_resource
def get_service():
    return FinancialClassifierService(models_dir=MODELS_DIR)

@st.cache_data
def load_json_files():
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics_data = json.load(f)
    with open(os.path.join(DATA_DIR, "eda_summary.json"), "r") as f:
        eda_summary = json.load(f)
    return metrics_data, eda_summary

service = get_service()
metrics_data, eda_summary = load_json_files()
best_model_name = metrics_data['best_model']
available_models = list(metrics_data['results'].keys())

# -----------------------------------------------------------------------------
# Sidebar Navigation & Project Meta
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="badge-tag">Case Study 135</div>', unsafe_allow_html=True)
    st.title("💳 FinClassify")
    st.caption("Personal Finance Behavioral Classifier")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "⚡ Live Classifier Studio",
            "📊 Model Comparative Study",
            "🔍 EDA & Spending Patterns",
            "📑 Academic Final Analysis"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎓 Academic Context")
    st.markdown("""
    - **Course:** B.Tech CSE Semester V
    - **Subject:** Machine Learning
    - **Dataset:** 20,000 Real Profiles ([Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits))
    - **Models:** 6 Algorithms Benchmarked
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn")

# -----------------------------------------------------------------------------
# 1. LIVE CLASSIFIER STUDIO
# -----------------------------------------------------------------------------
if app_mode == "⚡ Live Classifier Studio":
    st.markdown('<div class="badge-tag">Machine Learning Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Personal Finance Classification Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter financial metrics to classify behavior into <strong>Saver</strong>, <strong>Balanced</strong>, or <strong>High-Spender</strong> and receive personalized advisory recommendations.</p>', unsafe_allow_html=True)
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Analyzed Records", f"{eda_summary['total_records']:,}")
    with kpi2:
        st.metric("Peak Test Accuracy", f"{metrics_data['results'][best_model_name]['accuracy']}%")
    with kpi3:
        st.metric("Best Model", best_model_name)
    with kpi4:
        st.metric("Target Classes", "Saver | Balanced | Spender")
        
    st.markdown("---")
    
    # Presets bar
    st.subheader("⚡ Quick Archetype Presets")
    pcol1, pcol2, pcol3, _ = st.columns([1, 1, 1, 2])
    
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = {
            'income': 48000, 'expenses': 28000, 'savings': 20000,
            'loans': 1000, 'investments': 2000, 'cc_util': 20.0,
            'housing': 12000, 'food': 8000, 'transport': 3000,
            'health': 2000, 'ent': 1500, 'shop': 1500
        }
        
    if pcol1.button("🟢 Disciplined Saver", use_container_width=True):
        st.session_state.preset_data = {
            'income': 48000, 'expenses': 28000, 'savings': 20000,
            'loans': 500, 'investments': 2000, 'cc_util': 15.0,
            'housing': 12000, 'food': 8000, 'transport': 3000,
            'health': 2000, 'ent': 1500, 'shop': 1500
        }
    if pcol2.button("🔵 Balanced Professional", use_container_width=True):
        st.session_state.preset_data = {
            'income': 42000, 'expenses': 31000, 'savings': 11000,
            'loans': 2500, 'investments': 1500, 'cc_util': 38.0,
            'housing': 14000, 'food': 8500, 'transport': 3500,
            'health': 2000, 'ent': 1500, 'shop': 1500
        }
    if pcol3.button("🔴 High-Spender Lifestyle", use_container_width=True):
        st.session_state.preset_data = {
            'income': 40000, 'expenses': 36000, 'savings': 4000,
            'loans': 5000, 'investments': 1000, 'cc_util': 75.0,
            'housing': 16000, 'food': 10000, 'transport': 4000,
            'health': 2000, 'ent': 2000, 'shop': 2000
        }

    # Two column layout: Inputs on left, Results on right
    col_input, col_result = st.columns([1.1, 1])
    
    with col_input:
        st.markdown("### 📝 Enter Financial Profile")
        
        selected_model = st.selectbox(
            "Select Machine Learning Algorithm",
            available_models,
            index=available_models.index(best_model_name),
            help="Choose from 6 trained algorithms. Best model is highlighted by default."
        )
        
        st.markdown("#### Primary Monthly Cash Flow (₹/$)")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            income = st.number_input("Monthly Net Income", min_value=1000, max_value=2000000, 
                                     value=int(st.session_state.preset_data['income']), step=500)
            savings = st.number_input("Monthly Savings Deposit", min_value=0, max_value=2000000, 
                                      value=int(st.session_state.preset_data['savings']), step=500)
            loans = st.number_input("Monthly Loan / EMI Payments", min_value=0, max_value=500000, 
                                    value=int(st.session_state.preset_data['loans']), step=250)
        with fcol2:
            expenses = st.number_input("Total Monthly Expenses", min_value=500, max_value=2000000, 
                                       value=int(st.session_state.preset_data['expenses']), step=500)
            investments = st.number_input("Monthly Investments / Insurance", min_value=0, max_value=500000, 
                                          value=int(st.session_state.preset_data['investments']), step=250)
            cc_util = st.slider("Credit Card Utilization (%)", min_value=0.0, max_value=100.0, 
                                value=float(st.session_state.preset_data['cc_util']), step=1.0)
            
        st.markdown("#### Spending Allocation by Category (₹/$)")
        scol1, scol2 = st.columns(2)
        with scol1:
            housing = st.number_input("Housing & Utilities", min_value=0, value=int(st.session_state.preset_data['housing']), step=250)
            transport = st.number_input("Transportation", min_value=0, value=int(st.session_state.preset_data['transport']), step=100)
            entertainment = st.number_input("Entertainment & Leisure", min_value=0, value=int(st.session_state.preset_data['ent']), step=100)
        with scol2:
            food = st.number_input("Food & Dining", min_value=0, value=int(st.session_state.preset_data['food']), step=250)
            healthcare = st.number_input("Healthcare & Medical", min_value=0, value=int(st.session_state.preset_data['health']), step=100)
            shopping = st.number_input("Shopping & Miscellaneous", min_value=0, value=int(st.session_state.preset_data['shop']), step=100)

    # Execute Prediction
    input_payload = {
        'monthly_income': float(income),
        'monthly_expenses': float(expenses),
        'savings': float(savings),
        'loan_payments': float(loans),
        'investment_amount': float(investments),
        'housing_utilities': float(housing),
        'food_dining': float(food),
        'transportation': float(transport),
        'healthcare': float(healthcare),
        'entertainment': float(entertainment),
        'shopping_discretionary': float(shopping),
        'credit_card_utilization': float(cc_util)
    }
    
    result = service.predict(input_payload, model_name=selected_model)
    pred_cat = result['predicted_category']
    conf = result['confidence']
    health_score = result['health_score']
    metrics = result['metrics']
    recs = result['recommendations']
    alerts = result['alerts']
    
    with col_result:
        st.markdown("### 📊 Classification Output")
        
        # Archetype Banner
        if pred_cat == "Saver":
            st.markdown(f"""
            <div class="archetype-card-saver">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #34d399; font-weight: 600;">Predicted Archetype</div>
                <div style="font-size: 2rem; font-weight: 800; color: #10b981;">🟢 Saver</div>
                <div style="font-size: 0.9rem; color: #d1fae5; margin-top: 0.3rem;"><strong>{conf:.1f}% Confidence</strong> via {selected_model}</div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_cat == "Balanced":
            st.markdown(f"""
            <div class="archetype-card-balanced">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #60a5fa; font-weight: 600;">Predicted Archetype</div>
                <div style="font-size: 2rem; font-weight: 800; color: #3b82f6;">🔵 Balanced</div>
                <div style="font-size: 0.9rem; color: #dbeafe; margin-top: 0.3rem;"><strong>{conf:.1f}% Confidence</strong> via {selected_model}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="archetype-card-spender">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #f87171; font-weight: 600;">Predicted Archetype</div>
                <div style="font-size: 2rem; font-weight: 800; color: #f43f5e;">🔴 High-Spender</div>
                <div style="font-size: 0.9rem; color: #ffe4e6; margin-top: 0.3rem;"><strong>{conf:.1f}% Confidence</strong> via {selected_model}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Financial Health Meter
        st.markdown(f"**Financial Health Score:** `{health_score} / 100`")
        st.progress(health_score / 100.0)
        
        # Ratios 4-pack
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Savings Rate", f"{metrics['savings_rate']:.1f}%", help="Savings / Income (Ideal: >=20%)")
            st.metric("Debt-to-Income", f"{metrics['debt_to_income']:.1f}%", help="Debt Payments / Income (Ideal: <=20%)")
        with r2:
            st.metric("Expense Ratio", f"{metrics['expense_to_income']:.1f}%", help="Expenses / Income (Ideal: <=70%)")
            st.metric("Discretionary Ratio", f"{metrics['discretionary_ratio']:.1f}%", help="Wants / Total Expenses (Ideal: <=25%)")
            
        # Alerts
        if alerts:
            st.markdown("#### ⚠️ Vulnerability Warnings")
            for alert_text in alerts:
                st.markdown(f'<div class="alert-box">⚠️ {alert_text}</div>', unsafe_allow_html=True)
                
        # Spending Breakdown Chart
        st.markdown("#### 🍩 Spending Breakdown by Category")
        chart_data = pd.DataFrame({
            'Category': ['Housing & Utils', 'Food & Dining', 'Transportation', 'Healthcare', 'Entertainment', 'Shopping/Misc'],
            'Amount': [housing, food, transport, healthcare, entertainment, shopping]
        })
        donut = alt.Chart(chart_data).mark_arc(innerRadius=45).encode(
            theta=alt.Theta(field="Amount", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme='category20')),
            tooltip=['Category', 'Amount']
        ).properties(height=220)
        st.altair_chart(donut, use_container_width=True)
        
        # Benchmark 50/30/20 comparison chart
        st.markdown("#### 🎯 Actual vs 50/30/20 Rule Benchmark")
        b_comp = result['budget_comparison']
        bar_df = pd.DataFrame([
            {'Type': 'Needs (50%)', 'Allocation': 'Your Spending', 'Amount': b_comp['actual']['needs']},
            {'Type': 'Needs (50%)', 'Allocation': '50/30/20 Target', 'Amount': b_comp['benchmark']['needs']},
            {'Type': 'Wants (30%)', 'Allocation': 'Your Spending', 'Amount': b_comp['actual']['wants']},
            {'Type': 'Wants (30%)', 'Allocation': '50/30/20 Target', 'Amount': b_comp['benchmark']['wants']},
            {'Type': 'Savings (20%)', 'Allocation': 'Your Spending', 'Amount': b_comp['actual']['savings_investments']},
            {'Type': 'Savings (20%)', 'Allocation': '50/30/20 Target', 'Amount': b_comp['benchmark']['savings_investments']}
        ])
        bar_chart = alt.Chart(bar_df).mark_bar().encode(
            x=alt.X('Type:N', title=None),
            y=alt.Y('Amount:Q', title='Amount (₹/$)'),
            color=alt.Color('Allocation:N', scale=alt.Scale(domain=['Your Spending', '50/30/20 Target'], range=['#8b5cf6', '#4b5563'])),
            xOffset='Allocation:N',
            tooltip=['Type', 'Allocation', 'Amount']
        ).properties(height=220)
        st.altair_chart(bar_chart, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 💡 Tailored Financial Recommendations")
        for rec in recs:
            st.markdown(f"""
            <div class="rec-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #fff;">{rec['title']}</strong>
                    <span style="font-size: 0.75rem; background: rgba(99,102,241,0.2); color: #a5b4fc; padding: 2px 6px; border-radius: 4px;">{rec['tag']}</span>
                </div>
                <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 0.3rem;">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL COMPARATIVE STUDY
# -----------------------------------------------------------------------------
elif app_mode == "📊 Model Comparative Study":
    st.markdown('<div class="badge-tag">B.Tech CSE Case Study 135</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Model Comparative Study</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Empirical benchmarking of 6 supervised classification algorithms evaluated on the 20,000-record online dataset using 5-Fold Stratified Cross Validation.</p>', unsafe_allow_html=True)
    
    # Table of comparison
    st.subheader("📋 Comprehensive Model Performance Table")
    
    table_rows = []
    for mname, res in metrics_data['results'].items():
        table_rows.append({
            'Algorithm': mname,
            'Model Paradigm': 'Tree Ensemble' if ('Tree' in mname or 'Forest' in mname or 'Boosting' in mname) else ('Linear' if 'Logistic' in mname else ('Instance-Based' if 'Neighbors' in mname else 'Kernelized')),
            '5-Fold CV Accuracy': f"{res['cv_mean_accuracy']}% ± {res['cv_std']}%",
            'Test Accuracy': f"{res['accuracy']}%",
            'Macro Precision': f"{res['precision_macro']}%",
            'Macro Recall': f"{res['recall_macro']}%",
            'Macro F1-Score': f"{res['f1_macro']}%",
            'Weighted F1': f"{res['f1_weighted']}%",
            'Status': "★ Best Model" if mname == best_model_name else "Benchmarked"
        })
    df_models = pd.DataFrame(table_rows)
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    # Per-class table
    st.subheader("🎯 Per-Class Performance Breakdown (Saver | Balanced | Spender)")
    class_rows = []
    for mname, res in metrics_data['results'].items():
        pc = res['per_class']
        class_rows.append({
            'Algorithm': mname,
            'Saver Precision': f"{pc['Saver']['precision']}%",
            'Saver Recall': f"{pc['Saver']['recall']}%",
            'Saver F1': f"{pc['Saver']['f1']}%",
            'Balanced Precision': f"{pc['Balanced']['precision']}%",
            'Balanced Recall': f"{pc['Balanced']['recall']}%",
            'Balanced F1': f"{pc['Balanced']['f1']}%",
            'Spender Precision': f"{pc['High-Spender']['precision']}%",
            'Spender Recall': f"{pc['High-Spender']['recall']}%",
            'Spender F1': f"{pc['High-Spender']['f1']}%",
        })
    df_classes = pd.DataFrame(class_rows)
    st.dataframe(df_classes, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 Comparative Visualizations")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Accuracy & F1-Score Comparison")
        st.image(os.path.join(PLOTS_DIR, "model_comparison_bar.png"), use_container_width=True)
    with c2:
        st.markdown("#### 2x3 Confusion Matrices Grid")
        st.image(os.path.join(PLOTS_DIR, "confusion_matrices_grid.png"), use_container_width=True)
        
    st.markdown("#### Top 10 Feature Importances (Random Forest vs Gradient Boosting)")
    st.image(os.path.join(PLOTS_DIR, "feature_importances.png"), use_container_width=True)

# -----------------------------------------------------------------------------
# 3. EDA & SPENDING PATTERNS
# -----------------------------------------------------------------------------
elif app_mode == "🔍 EDA & Spending Patterns":
    st.markdown('<div class="badge-tag">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Financial Behavior Patterns & EDA</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-sub">Statistical profiling of <strong>{eda_summary["total_records"]:,} consumer records</strong> from the real Kaggle / GitHub Indian Personal Finance Dataset.</p>', unsafe_allow_html=True)
    
    # 3 Archetype Cards
    arch = eda_summary['archetypes']
    a1, a2, a3 = st.columns(3)
    
    with a1:
        st.markdown(f"""
        <div class="archetype-card-saver">
            <h3 style="color: #10b981; margin: 0 0 0.5rem 0;">🟢 Saver Profile</h3>
            <p style="color: #9ca3af; font-size: 0.85rem;"><strong>{arch['Saver']['count']:,} users ({arch['Saver']['pct_of_total']}%)</strong></p>
            <p><strong>Avg Income:</strong> ₹{arch['Saver']['avg_monthly_income']:,.2f}</p>
            <p><strong>Avg Expenses:</strong> ₹{arch['Saver']['avg_monthly_expenses']:,.2f}</p>
            <p><strong>Avg Savings:</strong> ₹{arch['Saver']['avg_savings']:,.2f}</p>
            <p><strong>Avg Savings Rate:</strong> <span style="color:#10b981; font-weight:700;">{arch['Saver']['avg_savings_rate']}%</span></p>
            <p><strong>Expense Ratio:</strong> {arch['Saver']['avg_expense_to_income']}%</p>
            <p><strong>Debt-to-Income:</strong> {arch['Saver']['avg_debt_to_income']}%</p>
            <p><strong>CC Utilization:</strong> {arch['Saver']['avg_cc_utilization']}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with a2:
        st.markdown(f"""
        <div class="archetype-card-balanced">
            <h3 style="color: #3b82f6; margin: 0 0 0.5rem 0;">🔵 Balanced Profile</h3>
            <p style="color: #9ca3af; font-size: 0.85rem;"><strong>{arch['Balanced']['count']:,} users ({arch['Balanced']['pct_of_total']}%)</strong></p>
            <p><strong>Avg Income:</strong> ₹{arch['Balanced']['avg_monthly_income']:,.2f}</p>
            <p><strong>Avg Expenses:</strong> ₹{arch['Balanced']['avg_monthly_expenses']:,.2f}</p>
            <p><strong>Avg Savings:</strong> ₹{arch['Balanced']['avg_savings']:,.2f}</p>
            <p><strong>Avg Savings Rate:</strong> <span style="color:#3b82f6; font-weight:700;">{arch['Balanced']['avg_savings_rate']}%</span></p>
            <p><strong>Expense Ratio:</strong> {arch['Balanced']['avg_expense_to_income']}%</p>
            <p><strong>Debt-to-Income:</strong> {arch['Balanced']['avg_debt_to_income']}%</p>
            <p><strong>CC Utilization:</strong> {arch['Balanced']['avg_cc_utilization']}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with a3:
        st.markdown(f"""
        <div class="archetype-card-spender">
            <h3 style="color: #f43f5e; margin: 0 0 0.5rem 0;">🔴 High-Spender Profile</h3>
            <p style="color: #9ca3af; font-size: 0.85rem;"><strong>{arch['High-Spender']['count']:,} users ({arch['High-Spender']['pct_of_total']}%)</strong></p>
            <p><strong>Avg Income:</strong> ₹{arch['High-Spender']['avg_monthly_income']:,.2f}</p>
            <p><strong>Avg Expenses:</strong> ₹{arch['High-Spender']['avg_monthly_expenses']:,.2f}</p>
            <p><strong>Avg Savings:</strong> ₹{arch['High-Spender']['avg_savings']:,.2f}</p>
            <p><strong>Avg Savings Rate:</strong> <span style="color:#f43f5e; font-weight:700;">{arch['High-Spender']['avg_savings_rate']}%</span></p>
            <p><strong>Expense Ratio:</strong> {arch['High-Spender']['avg_expense_to_income']}%</p>
            <p><strong>Debt-to-Income:</strong> {arch['High-Spender']['avg_debt_to_income']}%</p>
            <p><strong>CC Utilization:</strong> {arch['High-Spender']['avg_cc_utilization']}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("📊 Visual Exploratory Analysis")
    
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### 1. Target Class Frequency & Proportions")
        st.image(os.path.join(PLOTS_DIR, "eda_class_distribution.png"), use_container_width=True)
        st.markdown("#### 3. Category Spending Breakdown by Archetype")
        st.image(os.path.join(PLOTS_DIR, "eda_spending_breakdown.png"), use_container_width=True)
        st.markdown("#### 5. Monthly Income vs Monthly Expenses")
        st.image(os.path.join(PLOTS_DIR, "eda_income_vs_expense.png"), use_container_width=True)
        
    with e2:
        st.markdown("#### 2. Key Financial Ratios (Boxplots)")
        st.image(os.path.join(PLOTS_DIR, "eda_financial_ratios.png"), use_container_width=True)
        st.markdown("#### 4. Correlation Heatmap")
        st.image(os.path.join(PLOTS_DIR, "eda_correlation_matrix.png"), use_container_width=True)
        st.markdown("#### 6. Discretionary Spending vs Savings Rate")
        st.image(os.path.join(PLOTS_DIR, "eda_discretionary_vs_savings.png"), use_container_width=True)

# -----------------------------------------------------------------------------
# 4. ACADEMIC FINAL ANALYSIS
# -----------------------------------------------------------------------------
elif app_mode == "📑 Academic Final Analysis":
    st.markdown('<div class="badge-tag">Academic Report Documentation</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Academic Final Analysis & Findings</h1>', unsafe_allow_html=True)
    st.caption("Addressing all 5 core requirements of Section 7 (Semester V Machine Learning Case Study 135)")
    
    st.markdown("""
    ### 1. Important Financial Behavior Patterns
    - **Income Invariance:** Across the 20,000 real individual records, average incomes across Savers (₹42,037), Balanced (₹41,822), and High-Spenders (₹40,788) are virtually identical. Financial health is determined by **spending and debt allocation behavior** rather than raw income volume.
    - **Savings Rate Divergence:** Savers achieve an average savings rate of **36.66%**, compared to **26.28%** for Balanced individuals and **13.48%** for High-Spenders.
    - **Debt Obligation Escalation:** High-Spenders exhibit an average DTI of **11.53%** (nearly 26x higher than Savers at 0.44%), creating compounding interest penalties and reducing liquidity.
    
    ### 2. Best Classification Model
    - **Decision Tree**, **Random Forest**, and **Gradient Boosting** achieved top test accuracy (**99.98%**) and macro F1 (**99.98%**).
    - **Logistic Regression (99.62%)** demonstrates extraordinary linear separability in engineered ratio space with constant $O(1)$ inference latency and complete explainability.
    
    ### 3. Classification Accuracy
    - 5-Fold Stratified Cross-Validation on 16,000 training samples confirmed stable, non-overfitting performance (99.99% ± 0.02% for Decision Tree, 99.62% ± 0.12% for Logistic Regression).
    - Near-zero confusion matrix misclassification across all 4,000 held-out test records.
    
    ### 4. Real-World Limitations
    - **Cross-Sectional Assumption:** Current models analyze a single-month snapshot; cyclical or seasonal shocks (e.g., annual festivals or vacation travel) may temporarily skew archetype classification.
    - **Cost of Living Differences:** Regional disparities across Tier-1 vs Tier-3 cities require localized indexing.
    - **Self-Reported Veracity:** Deployed systems require Open Banking API integration to eliminate reporting inaccuracies.
    
    ### 5. Potential FinTech Applications
    - **Robo-Advisory Engines:** Tailoring automated portfolio allocations to user archetypes.
    - **Bank Early Delinquency Warning:** Pre-emptive credit risk underwriting detecting downward behavioral drift months before loan defaults occur.
    - **Smart Personal Financial Management (PFM) Mobile Apps:** Real-time push-notification interventions when mid-month discretionary spending crosses safe thresholds.
    - **Alternative Credit Scoring:** Enabling credit access for unbanked individuals by verifying disciplined saving habits.
    """)
    
    st.markdown("---")
    st.info("💡 You can run this Streamlit application locally using `streamlit run streamlit_app.py` or deploy directly to Streamlit Community Cloud (`share.streamlit.io`) by connecting your GitHub repository.")
