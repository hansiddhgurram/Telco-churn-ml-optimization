# 📊 Telco Customer Churn Prediction  
### End-to-End Machine Learning Project (Baseline → Optimized)

## 🔍 Project Overview
Customer churn is a critical problem for telecom companies, where retaining existing customers is often cheaper than acquiring new ones.  
This project builds an **end-to-end machine learning pipeline** to predict whether a customer will churn, while **systematically improving model performance** through preprocessing, feature engineering, model comparison, and hyperparameter tuning.

Unlike simple ML demos, this project emphasizes:
- **Baseline → Improvement → Optimization**
- **Clear reasoning behind each improvement**
- **Production-style code organization**

---

## 🎯 Problem Statement
Predict whether a telecom customer will **churn (leave the service)** based on demographic, service usage, contract, and billing information.

- **Type:** Binary Classification  
- **Target Variable:** `Churn` (Yes / No → 1 / 0)  
- **Primary Challenge:** Class imbalance and mixed feature types  

---

## 📁 Project Structure
telco-churn-ml-optimization/
│
├── data/
│ ├── raw/
│ │ └── telco_churn.csv # (not included in repo)
│ └── processed/
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_baseline_model.ipynb
│ ├── 03_preprocessing_improvements.ipynb
│ ├── 04_feature_engineering.ipynb
│ ├── 05_model_comparison.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│ └── 07_final_evaluation.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ └── evaluate.py
│
├── run_training.py
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Dataset
- **Name:** Telco Customer Churn  
- **Source:** Kaggle (IBM Analytics)  
- **Rows:** ~7,000  
- **Features:** Demographic, service usage, billing, and contract details  

📌 **Note:**  
The dataset is **not included** in this repository.  
Place it at:

---

## 🧠 Project Workflow

### 1️⃣ Data Understanding & EDA
- Explored churn distribution
- Identified class imbalance
- Detected data quality issues (`TotalCharges`)
- Analyzed churn vs tenure, contracts, and billing

📘 Notebook: `01_data_understanding.ipynb`

---

### 2️⃣ Baseline Model
- Logistic Regression
- Minimal preprocessing
- Naive categorical encoding

📉 Result:
- Accuracy reasonable
- **Recall for churn very low**

📘 Notebook: `02_baseline_model.ipynb`

---

### 3️⃣ Preprocessing Improvements
- One-hot encoding for categorical features
- Feature scaling
- Pipelines using `ColumnTransformer`

📈 Result:
- Improved recall and F1-score

📘 Notebook: `03_preprocessing_improvements.ipynb`

---

### 4️⃣ Feature Engineering & Class Imbalance
New features created:
- `AvgMonthlySpend`
- `TenureGroup`
- `HighMonthlyCharge`
- `ServiceCount`

Also applied:
- `class_weight='balanced'`

📈 Result:
- Significant recall improvement

📘 Notebook: `04_feature_engineering.ipynb`

---

### 5️⃣ Model Comparison
Compared:
- Logistic Regression
- Random Forest

📈 Result:
- Random Forest outperformed the linear model
- Better capture of non-linear customer behavior

📘 Notebook: `05_model_comparison.ipynb`

---

### 6️⃣ Hyperparameter Tuning
- GridSearchCV
- Optimized for **F1-score**
- Cross-validation to reduce overfitting

📈 Result:
- Best generalization performance

📘 Notebook: `06_hyperparameter_tuning.ipynb`

---

### 7️⃣ Final Evaluation
- Final performance comparison
- Confusion matrix analysis
- Business-focused conclusions

📘 Notebook: `07_final_evaluation.ipynb`

---

## 🧪 Final Model
**Tuned Random Forest Classifier**

Selected because:
- Highest F1-score
- Improved recall for churned customers
- Handles non-linear interactions
- Robust to mixed data types

---

## 📉 Evaluation Metrics
Due to class imbalance, **accuracy alone is misleading**.

Metrics used:
- Precision
- Recall
- F1-score
- Confusion Matrix

> In churn prediction, **false negatives are more costly than false positives**.

---

## 🧩 Modular Code (`src/`)
Core logic is refactored into reusable modules:

- `preprocessing.py` → data cleaning & feature engineering  
- `train.py` → model training pipelines  
- `evaluate.py` → evaluation utilities  

This improves:
- Maintainability
- Reproducibility
- Production readiness

---

## ▶️ How to Run the Project

### Option 1: Run Notebooks
```bash
jupyter notebook
Run notebooks in order (01 → 07)
python run_training.py
pip install -r requirements.txt

Key Learnings

ML performance improves incrementally
Feature engineering often matters more than model choice
Proper preprocessing significantly boosts results
Confusion matrix gives deeper insight than accuracy
Modular code improves real-world usability

Future Improvements

SHAP-based model explainability
Threshold optimization using business cost
Model persistence (joblib)
Deployment via Streamlit or REST API

Author

Hansiddh G
Machine Learning Enthusiast | Aspiring ML Engineer