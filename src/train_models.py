"""
Model Training and Comparative Study Module
Trains 6 Machine Learning Classifiers:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Support Vector Machine (SVM)

Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
Saves models, scalers, and evaluation metrics for deployment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Feature engineering
    df['savings_rate'] = (df['savings'] / df['monthly_income']) * 100
    df['expense_to_income'] = (df['monthly_expenses'] / df['monthly_income']) * 100
    df['debt_to_income'] = (df['loan_payments'] / df['monthly_income']) * 100
    df['investment_rate'] = (df['investment_amount'] / df['monthly_income']) * 100
    
    discretionary_spend = df['entertainment'] + df['shopping_discretionary']
    df['discretionary_ratio'] = (discretionary_spend / df['monthly_expenses']) * 100
    
    essential_spend = df['housing_utilities'] + df['food_dining'] + df['healthcare'] + df['transportation']
    df['essential_ratio'] = (essential_spend / df['monthly_expenses']) * 100
    
    feature_cols = [
        'monthly_income', 'monthly_expenses', 'savings', 'loan_payments', 'investment_amount',
        'housing_utilities', 'food_dining', 'transportation', 'healthcare',
        'entertainment', 'shopping_discretionary', 'credit_card_utilization',
        'savings_rate', 'expense_to_income', 'debt_to_income', 'investment_rate',
        'discretionary_ratio', 'essential_ratio'
    ]
    
    X = df[feature_cols].copy()
    y = df['financial_category'].copy()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'label_encoder': le,
        'classes': list(le.classes_)
    }

def get_models():
    return {
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42),
            'use_scaled': False,
            'description': 'Rule-based non-linear tree partitioned using Gini impurity with maximum interpretability.'
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
            'use_scaled': False,
            'description': 'Ensemble of bagging decision trees reducing variance and preventing overfitting.'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=3, random_state=42),
            'use_scaled': False,
            'description': 'Sequential boosting ensemble optimizing deviance loss via gradient-guided weak learners.'
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'use_scaled': True,
            'description': 'Multinomial linear classifier applying Softmax transformation with L2 regularization.'
        }
    }

def train_and_evaluate(data_dict, models_dir, reports_dir, web_plot_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(web_plot_dir, exist_ok=True)
    
    models = get_models()
    classes = data_dict['classes']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    fitted_models = {}
    confusion_matrices = {}
    
    print("=" * 80)
    print("TRAINING AND EVALUATING THE 4 BEST CLASSIFICATION MODELS")
    print("=" * 80)
    
    for name, config in models.items():
        clf = config['model']
        use_scaled = config['use_scaled']
        
        X_tr = data_dict['X_train_scaled'] if use_scaled else data_dict['X_train']
        X_te = data_dict['X_test_scaled'] if use_scaled else data_dict['X_test']
        
        # 5-Fold Cross Validation on Training Set
        cv_scores = cross_val_score(clf, X_tr, data_dict['y_train'], cv=cv, scoring='accuracy')
        
        # Fit on entire training set
        clf.fit(X_tr, data_dict['y_train'])
        fitted_models[name] = clf
        
        # Predict on Test Set
        y_pred = clf.predict(X_te)
        
        # Metrics
        acc = accuracy_score(data_dict['y_test'], y_pred)
        prec_macro = precision_score(data_dict['y_test'], y_pred, average='macro')
        rec_macro = recall_score(data_dict['y_test'], y_pred, average='macro')
        f1_macro = f1_score(data_dict['y_test'], y_pred, average='macro')
        
        prec_weighted = precision_score(data_dict['y_test'], y_pred, average='weighted')
        rec_weighted = recall_score(data_dict['y_test'], y_pred, average='weighted')
        f1_weighted = f1_score(data_dict['y_test'], y_pred, average='weighted')
        
        cm = confusion_matrix(data_dict['y_test'], y_pred)
        confusion_matrices[name] = cm
        
        # Per-class metrics
        prec_per_class = precision_score(data_dict['y_test'], y_pred, average=None)
        rec_per_class = recall_score(data_dict['y_test'], y_pred, average=None)
        f1_per_class = f1_score(data_dict['y_test'], y_pred, average=None)
        
        per_class_dict = {}
        for idx, cname in enumerate(classes):
            per_class_dict[cname] = {
                'precision': round(float(prec_per_class[idx]) * 100, 2),
                'recall': round(float(rec_per_class[idx]) * 100, 2),
                'f1': round(float(f1_per_class[idx]) * 100, 2)
            }
            
        results[name] = {
            'accuracy': round(float(acc) * 100, 2),
            'cv_mean_accuracy': round(float(cv_scores.mean()) * 100, 2),
            'cv_std': round(float(cv_scores.std()) * 100, 2),
            'precision_macro': round(float(prec_macro) * 100, 2),
            'recall_macro': round(float(rec_macro) * 100, 2),
            'f1_macro': round(float(f1_macro) * 100, 2),
            'precision_weighted': round(float(prec_weighted) * 100, 2),
            'recall_weighted': round(float(rec_weighted) * 100, 2),
            'f1_weighted': round(float(f1_weighted) * 100, 2),
            'confusion_matrix': cm.tolist(),
            'per_class': per_class_dict,
            'description': config['description'],
            'use_scaled': use_scaled
        }
        
        print(f"[{name}]")
        print(f"  Test Accuracy   : {acc*100:.2f}% (CV 5-Fold: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%)")
        print(f"  Macro Precision : {prec_macro*100:.2f}% | Recall: {rec_macro*100:.2f}% | F1: {f1_macro*100:.2f}%")
        print(f"  Weighted F1     : {f1_weighted*100:.2f}%\n")
        
        # Save individual model
        slug = name.lower().replace(' ', '_').replace('-', '_')
        joblib.dump(clf, os.path.join(models_dir, f"{slug}_model.joblib"))

    # Determine Best Model by F1-Score & Accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    print(f"★ Best Performing Model: {best_model_name} (F1-Macro: {results[best_model_name]['f1_macro']}%)")
    
    # Save Scaler, LabelEncoder, Feature Names, and Metadata
    joblib.dump(data_dict['scaler'], os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(data_dict['label_encoder'], os.path.join(models_dir, "label_encoder.joblib"))
    
    metadata = {
        'best_model': best_model_name,
        'feature_cols': data_dict['feature_cols'],
        'classes': classes,
        'results': results
    }
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate Comparison Visualizations
    generate_evaluation_plots(results, confusion_matrices, classes, fitted_models, data_dict, reports_dir, web_plot_dir)
    
    return metadata

def generate_evaluation_plots(results, confusion_matrices, classes, fitted_models, data_dict, reports_dir, web_plot_dir):
    def save_both(fig, filename):
        fig.savefig(os.path.join(reports_dir, filename), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(web_plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved evaluation figure: {filename}")
        
    model_names = list(results.keys())
    
    # 1. Comparative Accuracy & F1 Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.22
    
    accs = [results[m]['accuracy'] for m in model_names]
    precs = [results[m]['precision_macro'] for m in model_names]
    recs = [results[m]['recall_macro'] for m in model_names]
    f1s = [results[m]['f1_macro'] for m in model_names]
    
    r1 = ax.bar(x - 1.5 * width, accs, width, label='Accuracy', color='#3b82f6')
    r2 = ax.bar(x - 0.5 * width, precs, width, label='Precision (Macro)', color='#10b981')
    r3 = ax.bar(x + 0.5 * width, recs, width, label='Recall (Macro)', color='#f59e0b')
    r4 = ax.bar(x + 1.5 * width, f1s, width, label='F1-Score (Macro)', color='#8b5cf6')
    
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Comparative Performance Analysis Across the 4 Best Classification Algorithms', fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontweight='medium')
    ax.set_ylim(80, 102)
    ax.legend(loc='lower right', frameon=True)
    
    # Add values on top of F1 bars
    for bar in r4:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                    
    save_both(fig, "model_comparison_bar.png")
    
    # 2. Confusion Matrices Grid (2x2 for 4 models)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f"{name}\nAcc: {results[name]['accuracy']}% | F1: {results[name]['f1_macro']}%",
                     fontweight='bold', pad=8)
        ax.set_xlabel('Predicted Label', fontweight='medium')
        ax.set_ylabel('True Label', fontweight='medium')
        
    plt.tight_layout()
    save_both(fig, "confusion_matrices_grid.png")
    
    # 3. Feature Importances (Random Forest & Gradient Boosting)
    rf_model = fitted_models.get('Random Forest')
    gb_model = fitted_models.get('Gradient Boosting')
    
    if rf_model and gb_model:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        feature_names = data_dict['feature_cols']
        
        # RF
        importances_rf = rf_model.feature_importances_
        indices_rf = np.argsort(importances_rf)[::-1]
        sns.barplot(x=importances_rf[indices_rf][:10], y=[feature_names[i] for i in indices_rf][:10],
                    palette='viridis', ax=axes[0])
        axes[0].set_title("Top 10 Feature Importances (Random Forest)", fontweight='bold')
        axes[0].set_xlabel("Relative Importance")
        
        # GB
        importances_gb = gb_model.feature_importances_
        indices_gb = np.argsort(importances_gb)[::-1]
        sns.barplot(x=importances_gb[indices_gb][:10], y=[feature_names[i] for i in indices_gb][:10],
                    palette='mako', ax=axes[1])
        axes[1].set_title("Top 10 Feature Importances (Gradient Boosting)", fontweight='bold')
        axes[1].set_xlabel("Relative Importance")
        
        plt.tight_layout()
        save_both(fig, "feature_importances.png")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "personal_finance_data.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports", "figures")
    web_plot_dir = os.path.join(base_dir, "static", "plots")
    
    data = prepare_data(csv_file)
    print(f"Training dataset size: {len(data['X_train'])} | Test set size: {len(data['X_test'])}")
    print(f"Target classes: {data['classes']}")
    
    metadata = train_and_evaluate(data, models_dir, reports_dir, web_plot_dir)
    print("\nTraining completed successfully! All artifacts exported.")
