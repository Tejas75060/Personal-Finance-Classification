"""
Exploratory Data Analysis (EDA) Module
Analyzes personal financial data and generates visualizations & statistics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'figure.autolayout': True,
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif'
})

PALETTE = {
    'Saver': '#10b981',       # Emerald green
    'Balanced': '#3b82f6',    # Royal blue
    'High-Spender': '#ef4444' # Coral red
}

def load_and_engineer_features(csv_path):
    df = pd.read_csv(csv_path)
    
    # Financial domain ratio engineering
    df['savings_rate'] = (df['savings'] / df['monthly_income']) * 100
    df['expense_to_income'] = (df['monthly_expenses'] / df['monthly_income']) * 100
    df['debt_to_income'] = (df['loan_payments'] / df['monthly_income']) * 100
    df['investment_rate'] = (df['investment_amount'] / df['monthly_income']) * 100
    
    # Discretionary spending: Entertainment + Shopping
    discretionary_spend = df['entertainment'] + df['shopping_discretionary']
    df['discretionary_ratio'] = (discretionary_spend / df['monthly_expenses']) * 100
    
    # Essential spending: Housing + Food + Healthcare + Transport
    essential_spend = df['housing_utilities'] + df['food_dining'] + df['healthcare'] + df['transportation']
    df['essential_ratio'] = (essential_spend / df['monthly_expenses']) * 100
    
    return df

def generate_eda_visualizations(df, output_dir, web_plot_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(web_plot_dir, exist_ok=True)
    
    saved_plots = []
    
    def save_fig(fig, filename):
        path1 = os.path.join(output_dir, filename)
        path2 = os.path.join(web_plot_dir, filename)
        fig.savefig(path1, dpi=300, bbox_inches='tight')
        fig.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots.append(filename)
        print(f"Saved: {filename}")
        
    # 1. Target Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    class_counts = df['financial_category'].value_counts()
    colors = [PALETTE[cat] for cat in class_counts.index]
    
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=PALETTE, ax=axes[0])
    axes[0].set_title("Class Frequency Distribution", fontweight='bold', pad=12)
    axes[0].set_ylabel("Number of Users")
    axes[0].set_xlabel("Financial Category")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 25, f"{v} ({v/len(df)*100:.1f}%)", ha='center', fontweight='semibold')
        
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=140, explode=[0.03, 0.03, 0.03],
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title("Class Proportion Breakdown", fontweight='bold', pad=12)
    save_fig(fig, "eda_class_distribution.png")
    
    # 2. Key Financial Ratios by Category (Boxplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ratios = [
        ('savings_rate', 'Savings Rate (% of Income)', axes[0, 0]),
        ('expense_to_income', 'Expense-to-Income Ratio (%)', axes[0, 1]),
        ('debt_to_income', 'Debt-to-Income Ratio (DTI %)', axes[1, 0]),
        ('discretionary_ratio', 'Discretionary Spend (% of Expenses)', axes[1, 1])
    ]
    for col, title, ax in ratios:
        sns.boxplot(x='financial_category', y=col, data=df, palette=PALETTE, ax=ax, width=0.5, boxprops=dict(alpha=0.85))
        sns.stripplot(x='financial_category', y=col, data=df, color='black', alpha=0.08, jitter=0.2, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("%")
    save_fig(fig, "eda_financial_ratios.png")
    
    # 3. Average Spending Category Breakdown by Class
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = ['housing_utilities', 'food_dining', 'transportation', 'healthcare', 'entertainment', 'shopping_discretionary']
    cat_names = ['Housing & Utils', 'Food & Dining', 'Transportation', 'Healthcare', 'Entertainment', 'Shopping']
    
    cat_means = df.groupby('financial_category')[categories].mean()
    # Normalize to percentage of total mean expenses per class
    cat_pcts = cat_means.div(cat_means.sum(axis=1), axis=0) * 100
    
    cat_pcts.loc[['Saver', 'Balanced', 'High-Spender']].plot(
        kind='bar', stacked=True, ax=ax, colormap='Spectral', edgecolor='white', linewidth=0.8
    )
    ax.set_title("Spending Category Composition (% of Expenses by Financial Archetype)", fontweight='bold', pad=12)
    ax.set_ylabel("Share of Total Monthly Spending (%)")
    ax.set_xlabel("Financial Archetype")
    ax.legend(cat_names, bbox_to_anchor=(1.02, 1), loc='upper left', title="Categories")
    plt.xticks(rotation=0)
    save_fig(fig, "eda_spending_breakdown.png")
    
    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    num_cols = [
        'monthly_income', 'monthly_expenses', 'savings', 'loan_payments',
        'investment_amount', 'credit_card_utilization', 'savings_rate',
        'expense_to_income', 'debt_to_income', 'discretionary_ratio'
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Financial Indicators", fontweight='bold', pad=14)
    save_fig(fig, "eda_correlation_matrix.png")
    
    # 5. Income vs Monthly Expenses Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='monthly_income', y='monthly_expenses', hue='financial_category',
        data=df, palette=PALETTE, alpha=0.65, s=40, edgecolor='none', ax=ax
    )
    # Add 45-degree break-even line
    max_val = min(df['monthly_income'].max(), df['monthly_expenses'].max())
    ax.plot([2000, 20000], [2000, 20000], 'k--', alpha=0.5, label='Break-even (Expenses = Income)')
    ax.set_title("Monthly Income vs. Monthly Expenses", fontweight='bold', pad=12)
    ax.set_xlabel("Monthly Income ($)")
    ax.set_ylabel("Monthly Expenses ($)")
    ax.legend(title="Financial Archetype")
    save_fig(fig, "eda_income_vs_expense.png")
    
    # 6. Discretionary Ratio vs Savings Rate (Archetype Clustering)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='discretionary_ratio', y='savings_rate', hue='financial_category',
        data=df, palette=PALETTE, alpha=0.7, s=45, edgecolor='none', ax=ax
    )
    ax.set_title("Discretionary Spending Ratio vs. Savings Rate", fontweight='bold', pad=12)
    ax.set_xlabel("Discretionary Spending (% of Expenses)")
    ax.set_ylabel("Savings Rate (% of Monthly Income)")
    ax.axhline(20, color='gray', linestyle=':', alpha=0.6, label='20% Savings Benchmark')
    ax.axvline(30, color='gray', linestyle='--', alpha=0.6, label='30% Discretionary Benchmark')
    ax.legend(title="Financial Archetype")
    save_fig(fig, "eda_discretionary_vs_savings.png")
    
    return saved_plots

def compute_eda_summary(df):
    archetype_stats = {}
    for cat in ['Saver', 'Balanced', 'High-Spender']:
        sub = df[df['financial_category'] == cat]
        archetype_stats[cat] = {
            'count': int(len(sub)),
            'pct_of_total': round(len(sub) / len(df) * 100, 1),
            'avg_monthly_income': round(float(sub['monthly_income'].mean()), 2),
            'avg_monthly_expenses': round(float(sub['monthly_expenses'].mean()), 2),
            'avg_savings': round(float(sub['savings'].mean()), 2),
            'avg_savings_rate': round(float(sub['savings_rate'].mean()), 2),
            'avg_expense_to_income': round(float(sub['expense_to_income'].mean()), 2),
            'avg_debt_to_income': round(float(sub['debt_to_income'].mean()), 2),
            'avg_investment_rate': round(float(sub['investment_rate'].mean()), 2),
            'avg_discretionary_ratio': round(float(sub['discretionary_ratio'].mean()), 2),
            'avg_cc_utilization': round(float(sub['credit_card_utilization'].mean()), 1)
        }
        
    overall_summary = {
        'total_records': len(df),
        'num_features': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'median_income': round(float(df['monthly_income'].median()), 2),
        'median_expenses': round(float(df['monthly_expenses'].median()), 2),
        'archetypes': archetype_stats
    }
    return overall_summary

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "personal_finance_data.csv")
    plots_dir = os.path.join(base_dir, "reports", "figures")
    web_plots_dir = os.path.join(base_dir, "static", "plots")
    
    df = load_and_engineer_features(csv_file)
    print("Dataset loaded and features engineered. Total shape:", df.shape)
    
    plots = generate_eda_visualizations(df, plots_dir, web_plots_dir)
    print(f"Generated {len(plots)} EDA plots.")
    
    summary = compute_eda_summary(df)
    summary_path = os.path.join(base_dir, "data", "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("EDA Summary exported to:", summary_path)
