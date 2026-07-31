# 💼 Salary Prediction - Machine Learning Project

An end-to-end Machine Learning project built for predicting employee salaries based on demographic and professional features (Age, Gender, Education Level, Job Title, Years of Experience). Designed following clean, internship-ready ML standards.

---

## 📊 Model Comparison Results

| Rank | Model | Train R² | Test R² | MAE ($) | RMSE ($) | 5-Fold CV R² |
|---|---|---|---|---|---|---|
| **1 🏆** | **Tuned XGBoost** | **0.9642** | **0.8872** | **$10,450.50** | **$14,597.43** | **0.8852** |
| 2 | Tuned Random Forest | 0.9779 | 0.8810 | $10,501.12 | $14,988.87 | 0.8784 |
| 3 | Random Forest (Baseline) | 0.9833 | 0.8764 | $10,588.21 | $15,280.53 | 0.8774 |
| 4 | Linear Regression | 0.9866 | 0.8670 | $12,303.90 | $15,851.15 | 0.8257 |
| 5 | XGBoost (Baseline) | 0.9956 | 0.8392 | $11,533.67 | $17,431.45 | 0.8795 |
| 6 | Decision Tree | 0.9984 | 0.8323 | $12,000.00 | $17,802.33 | 0.8373 |

---

## 🛠 Features & Preprocessing

- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore')` applied to `Gender`, `Education Level`, and `Job Title`.
- **Numerical Scaling**: `StandardScaler()` applied to `Age` and `Years of Experience`.
- **Leakage Prevention**: Built using Scikit-Learn `Pipeline` and `ColumnTransformer`.

---

## 📁 Repository Structure

```
salary-prediction/
│
├── dataset/
│   └── Salary Data.csv                         # Raw dataset
│
├── notebooks/
│   ├── EDA.ipynb                               # Exploratory Data Analysis
│   └── Model_Training_and_Optimization.ipynb   # Internship Model Training Pipeline
│
├── models/                                     # Saved Models
│   ├── best_model.pkl                          # Saved best pipeline (Tuned XGBoost)
│   └── metrics.json                            # Evaluation summary metrics
│
├── venv/                                       # Python Virtual Environment
└── README.md                                   # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run Notebooks
Open Jupyter Notebook or VS Code:
- [`notebooks/EDA.ipynb`](file:///c:/Users/lenovo/OneDrive/Desktop/AI-ML-Projects/salary-prediction/notebooks/EDA.ipynb)
- [`notebooks/Model_Training_and_Optimization.ipynb`](file:///c:/Users/lenovo/OneDrive/Desktop/AI-ML-Projects/salary-prediction/notebooks/Model_Training_and_Optimization.ipynb)

### 3. Load & Predict in Python
```python
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load('models/best_model.pkl')

# Sample input
sample_data = pd.DataFrame([{
    'Age': 30,
    'Gender': 'Male',
    'Education Level': "Master's",
    'Job Title': 'Data Scientist',
    'Years of Experience': 4
}])

predicted_salary = model.predict(sample_data)
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
```
