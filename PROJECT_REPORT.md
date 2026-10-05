# B.Tech CSE Semester V — Machine Learning Case Study 135
# Personal Finance Classification Using Machine Learning

---

## 1. Title
**Personal Finance Classification Using Machine Learning**  
- **Course:** B.Tech Computer Science & Engineering (2024-28), Semester V  
- **Subject:** Machine Learning (Case Study / Problem Statement 135)  
- **Dataset Source:** Real-world [Indian Personal Finance and Spending Habits Dataset (Kaggle & GitHub)](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits)  
- **Observations:** 20,000 real individual financial records  
- **Target Classes:** `Saver` | `Balanced` | `High-Spender`  

---

## 2. Problem Statement
A modern personal-finance platform seeks to automatically classify users based on their multi-dimensional financial behavior, such as spending habits, saving rates, income levels, debt commitments, and transaction category allocations. 

The primary objective is to empower individuals to understand their behavioral archetype, identify structural cash-flow vulnerabilities, and receive data-driven, actionable financial management recommendations to build long-term financial resilience.

---

## 3. Objectives
1. **Analyze personal financial data** across income, expenditure, debt obligations, savings, and category-level allocations using a real online consumer dataset of 20,000 observations.
2. **Identify spending and saving patterns** that differentiate Savers, Balanced spenders, and High-Spenders.
3. **Perform preprocessing and Exploratory Data Analysis (EDA)** with rich statistical visualizations.
4. **Build the four best classification models:**
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
5. **Conduct a comparative performance study** evaluating Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), and Confusion Matrices.
6. **Deploy the classification system** through an interactive, production-ready web application providing real-time behavioral classification, health scoring, and personalized financial advisory.
7. **Document final analysis:** identifying important behavioral patterns, best model selection, classification accuracy, real-world limitations, and commercial FinTech applications.

---

## 4. Dataset Source & Financial Feature Engineering

### 4.1 Online Dataset Origin
The dataset was directly ingested from the open-source **Indian Personal Finance and Spending Habits Dataset** hosted on Kaggle and GitHub:
- **Kaggle Source:** `https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits`
- **GitHub Repository:** `https://github.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits`
- **Raw File URL:** `https://raw.githubusercontent.com/Somnath-Fintech/Indian-Personal-Finance-and-Spending-Habits/main/Dataset/data.csv`
- **Sample Size:** 20,000 records, 27 original features

### 4.2 Standardized Features
- `monthly_income`: Gross/Net monthly earning (₹/$)
- `monthly_expenses`: Total monthly living expenses (Sum of all living expense categories)
- `savings`: Monthly surplus savings cash flow
- `loan_payments`: Total monthly debt/EMI obligations
- `investment_amount`: Monthly insurance / long-term investment protection
- `housing_utilities`: Shelter, rent/mortgage, utilities
- `food_dining`: Groceries, dining out, food delivery
- `transportation`: Fuel, public transit, commute
- `healthcare`: Medical expenses, wellness
- `entertainment`: Leisure, hobbies, recreation
- `shopping_discretionary`: Education and miscellaneous discretionary spending
- `credit_card_utilization`: Revolving credit balance as % of limit

### 4.3 Engineered Financial Domain Ratios
1. **Savings Rate:**
   $$\text{Savings Rate} = \left(\frac{\text{Savings}}{\text{Monthly Income}}\right) \times 100$$
2. **Expense-to-Income Ratio:**
   $$\text{Expense Ratio} = \left(\frac{\text{Monthly Expenses}}{\text{Monthly Income}}\right) \times 100$$
3. **Debt-to-Income Ratio (DTI):**
   $$\text{DTI} = \left(\frac{\text{Loan Payments}}{\text{Monthly Income}}\right) \times 100$$
4. **Investment Rate:**
   $$\text{Investment Rate} = \left(\frac{\text{Investment Amount}}{\text{Monthly Income}}\right) \times 100$$
5. **Discretionary Spending Ratio:**
   $$\text{Discretionary Ratio} = \left(\frac{\text{Entertainment} + \text{Shopping}}{\text{Monthly Expenses}}\right) \times 100$$
6. **Essential Spending Ratio:**
   $$\text{Essential Ratio} = \left(\frac{\text{Housing} + \text{Food} + \text{Healthcare} + \text{Transport}}{\text{Monthly Expenses}}\right) \times 100$$

---

## 5. Exploratory Data Analysis & Behavioral Profiling

### 5.1 Archetype Statistical Summary (N = 20,000)
| Metric | Saver Archetype | Balanced Archetype | High-Spender Archetype |
| :--- | :---: | :---: | :---: |
| **Sample Count** | 5,625 (28.1%) | 8,627 (43.1%) | 5,748 (28.7%) |
| **Mean Monthly Income** | ₹42,037.16 | ₹41,822.12 | ₹40,788.36 |
| **Mean Monthly Expenses** | ₹26,639.50 | ₹30,840.85 | ₹35,290.77 |
| **Mean Savings Rate** | **36.66%** | **26.28%** | **13.48%** |
| **Mean Expense-to-Income** | **63.34%** | **73.72%** | **86.56%** |
| **Mean Debt-to-Income (DTI)** | **0.44%** | **3.60%** | **11.53%** |
| **Mean Investment Rate** | 3.37% | 3.50% | 3.60% |
| **Mean Credit Card Utilization** | **38.3%** | **47.2%** | **61.2%** |

### 5.2 Key Behavioral Insights
1. **Income Invariance:** Average incomes across the three archetypes are virtually identical (~₹41,000–₹42,000/month). Financial health is determined by *cash allocation behavior* rather than raw earning capacity.
2. **Savings Divergence:** Savers maintain an average savings rate of 36.66%, compared to 26.28% for Balanced individuals and just 13.48% for High-Spenders.
3. **Debt-to-Income Escalation:** High-Spenders exhibit an average DTI of 11.53% (nearly 26x higher than Savers at 0.44%), demonstrating how debt obligations choke cash reserves.

---

## 6. Machine Learning Algorithms
The four best classification models spanning rule-based, ensemble, and linear paradigms were selected, implemented, and tuned:

1. **Decision Tree:** Non-linear rule-based tree partitioned via Gini Impurity with max depth regularization ($d=6$). Provides near-instant inference and complete tree path explainability.
2. **Random Forest:** Ensemble bagging model aggregating 100 bootstrapped decision trees to minimize prediction variance and capture multi-feature interactions.
3. **Gradient Boosting:** Sequential boosting ensemble optimizing multi-class deviance loss with 80 estimators and learning rate $\eta=0.1$.
4. **Logistic Regression:** Multinomial linear classifier with Softmax activation and L2 regularization ($C=1.0$), delivering linear separability in engineered ratio space.

---

## 7. Comparative Performance Study

The models were evaluated using **5-Fold Stratified Cross-Validation** on the training partition ($N=16,000$) and tested on an independent hold-out test set ($N=4,000$).

### 7.1 Performance Comparison Table
| Algorithm | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** ★ | **99.99% ± 0.02%** | **99.98%** | **99.97%** | **99.98%** | **99.98%** | **99.98%** |
| **Random Forest** | 99.99% ± 0.02% | **99.98%** | 99.97% | 99.98% | **99.98%** | 99.98% |
| **Gradient Boosting** | 99.99% ± 0.02% | **99.98%** | 99.97% | 99.98% | **99.98%** | 99.98% |
| **Logistic Regression** | 99.62% ± 0.12% | 99.62% | 99.65% | 99.62% | 99.64% | 99.63% |

### 7.2 Confusion Matrix Breakdown
Confusion matrices across the models demonstrated exceptional discriminative ability:
- **Decision Tree & Tree Ensembles:** Misclassified only 1 out of 4,000 test records, achieving near 100% precision and recall across all three classes.
- **Logistic Regression:** Achieved 99.62% test accuracy, misclassifying only 15 out of 4,000 cases, highlighting near-perfect linear separability in engineered ratio space.
- **Saver Class Isolation:** All models exhibit near-perfect recognition ($>99.6\%$ precision and recall). Savers maintain high savings rates and low expense ratios, forming an unambiguous boundary.

---

## 8. Deployment System & Architecture

The classification system is deployed as an interactive, production-ready web application via **Streamlit Cloud**:
1. **Interactive Frontend & Dashboards:** Built with Streamlit (`streamlit_app.py`), offering real-time user profile entry, preset scenario loaders (Saver, Balanced, High-Spender), dynamic Altair budget allocation charts, and comparative metric tabs.
2. **Inference Engine:** `FinancialClassifierService` executing real-time feature derivation, ratio computations, standard scaling, multi-model prediction, 0-100 financial health scoring, and rule-based advisory generation.
3. **Cloud Resilience:** Autonomous in-environment fallback mechanism ensuring cross-version compatibility between scikit-learn model artifacts and Python runtimes on Streamlit Cloud.
4. **Live URL:** [https://personal-finance-classification-kgf4ksx4x4nbbg2dssj6nr.streamlit.app/](https://personal-finance-classification-kgf4ksx4x4nbbg2dssj6nr.streamlit.app/)

---

## 9. Final Analysis

### 9.1 Important Financial Behavior Patterns
- **Savings Rate & Expense Ratio** are the single most influential determinants of financial classification (over 70% relative feature importance in Random Forest and Decision Tree).
- **Debt Service Burdens** compound cash-flow instability when DTI exceeds 10% in real consumer households.
- **Behavior over Income:** Gross salary does not predict savings discipline; disciplined allocation does.

### 9.2 Best Classification Model
- **Decision Tree**, **Random Forest**, and **Gradient Boosting** achieved top test accuracy (**99.98%**) and macro F1 (**99.98%**).
- For enterprise deployment, **Logistic Regression (99.62%)** and **Decision Tree (99.98%)** offer ideal computational performance with constant $O(1)$ inference latency and complete explainability.

### 9.3 Limitations
- **Cross-Sectional Assumption:** Current models analyze a single-month snapshot; cyclical or seasonal shocks (e.g., annual festivals or vacation travel) may temporarily skew archetype classification.
- **Cost of Living Differences:** Regional disparities across Tier-1 vs Tier-3 cities require localized indexing.
- **Self-Reported Veracity:** Deployed systems require Open Banking API integration to eliminate reporting inaccuracies.

### 9.4 Potential Applications
1. **FinTech Robo-Advisors:** Automated portfolio construction dynamically calibrated to user archetypes.
2. **Early Delinquency Warning Systems:** Banking platforms can identify deteriorating spending velocity months before loan default.
3. **Smart Budgeting Applications:** Micro-nudges alerting users when discretionary spend crosses monthly limits.
4. **Alternative Credit Scoring:** Behavioral credit profiling for under-banked individuals lacking traditional bureau scores.
