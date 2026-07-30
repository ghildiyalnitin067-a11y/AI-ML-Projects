# Student Mental Health & CGPA Analysis: Predictive Modeling & Streamlit Web App

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2C3E50?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

An end-to-end Data Science and Machine Learning project analyzing the relationship between academic performance (CGPA), student demographics, and mental health conditions (Depression, Anxiety, Panic Attacks). Features a production-ready **Streamlit Web Application**, hyperparameter-tuned machine learning classifiers, and a dedicated **Engineering Student Cohort Analysis**.

---

## Technical Highlights & Key Features

- **Automated Data Engineering Pipeline**: Automated preprocessing addressing missing values, string normalization, CGPA range ordinal mapping (`0-1.99` to `3.50-4.00`), and binary variable encoding.
- **Interactive Streamlit Web App (`app.py`)**: Live dashboard featuring interactive EDA charts, cohort filtering (All Students vs. Engineering Majors), and a real-time **Depression Risk Predictor**.
- **Multi-Algorithm Optimization Suite**: GridSearch hyperparameter tuning across **Random Forest, Decision Trees, XGBoost, Gradient Boosting (GBM), Logistic Regression (L1/L2), SVM, and Stochastic Gradient Descent (SGD)** with 5-fold Stratified Cross-Validation.
- **Specialized Engineering Analysis (`engineering_student_mental_health_analysis.ipynb`)**: Cohort study analyzing academic pressure and mental health indicators specifically among Engineering students.
- **Cloud Deployment Ready**: Cross-platform path resolution, automatic model training fallback on startup, `.streamlit/config.toml`, and `Procfile` included.

---

## Repository Structure

```
student-mental-health-on-cgpa/
├── dataset/
│   ├── Student Mental health.csv                 # Raw survey dataset
│   └── cleaned_student_mental_health.csv           # Cleaned feature-engineered dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                      # Data cleaning & feature pipeline
│   └── train_model.py                             # ML training & GridSearch tuning pipeline
├── models/
│   ├── best_model.pkl                            # Serialized optimal ML classifier
│   └── scaler.pkl                                # Fitted StandardScaler artifact
├── .streamlit/
│   └── config.toml                               # Streamlit headless server config & theme
├── app.py                                         # Interactive Streamlit Web Application
├── student_mental_health_analysis.ipynb           # Jupyter Notebook (Full Dataset)
├── engineering_student_mental_health_analysis.ipynb # Jupyter Notebook (Engineering Cohort)
├── Procfile                                       # Deployment configuration for web hosts
├── requirements.txt                               # Project dependencies
├── .gitignore                                     # Environment & cache exclusions
└── README.md                                      # GitHub project documentation
```

---

## Dataset Overview

- **Source**: Higher education student survey dataset.
- **Features**:
  - `gender`: Male / Female (encoded binary)
  - `age`: Student age (imputed with integer median)
  - `course`: Degree program / Major (normalized text)
  - `year_of_study`: Academic year (1 to 4)
  - `cgpa`: Academic performance bands (`0-1.99`, `2.00-2.49`, `2.50-2.99`, `3.00-3.49`, `3.50-4.00`)
  - `marital_status`: Binary Yes/No
  - `anxiety`, `panic_attack`: Secondary mental health indicators
- **Target Variable**: `depression` (Binary Classification: 0 = Low Risk, 1 = High Risk)

---

## Model Benchmark & Evaluation Summary

All models were evaluated on an 80-20 stratified test split using **5-Fold Stratified Cross-Validation** and **GridSearchCV** hyperparameter tuning:

| Model | Test Accuracy | Precision | Recall | F1-Score | Best Hyperparameters |
|---|---|---|---|---|---|
| **Random Forest Classifier** | **0.8462** | **1.0000** | **0.5556** | **0.7143** | `{'max_depth': 5, 'n_estimators': 25}` |
| **Decision Tree Classifier** | 0.8077 | 0.7500 | 0.6667 | 0.7059 | `{'criterion': 'entropy', 'max_depth': 7}` |
| **Gradient Boosting (GBM)** | 0.7692 | 0.7143 | 0.5556 | 0.6250 | `{'learning_rate': 0.1, 'max_depth': 3}` |
| **Support Vector Machine (SVM)** | 0.7692 | 0.7143 | 0.5556 | 0.6250 | `{'C': 10.0, 'kernel': 'rbf'}` |
| **XGBoost Classifier** | 0.8077 | 1.0000 | 0.4444 | 0.6154 | `{'learning_rate': 0.05, 'max_depth': 3}` |
| **Logistic Regression (L1/L2)** | 0.8077 | 1.0000 | 0.4444 | 0.6154 | `{'C': 1.0, 'solver': 'lbfgs'}` |
| **Stochastic Gradient Descent (SGD)**| 0.7692 | 0.8000 | 0.4444 | 0.5714 | `{'alpha': 0.001, 'penalty': 'elasticnet'}` |

---

## Installation & Local Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/student-mental-health-on-cgpa.git
   cd student-mental-health-on-cgpa
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # On Windows PowerShell:
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Execute Data Cleaning Pipeline
```bash
python -m src.data_preprocessing
```

### 2. Execute ML Training & GridSearch Optimization
```bash
python -m src.train_model
```

### 3. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```
*Access the web app at `http://localhost:8501` in your browser.*

### 4. Open Interactive Jupyter Notebooks
```bash
jupyter notebook student_mental_health_analysis.ipynb
# or for Engineering Cohort:
jupyter notebook engineering_student_mental_health_analysis.ipynb
```

---

## Cloud Deployment Guide

### Deploying to Streamlit Community Cloud (Free)

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Complete student mental health ML app and Streamlit dashboard"
   git push origin main
   ```
2. Log into **[share.streamlit.io](https://share.streamlit.io)** with GitHub.
3. Click **New App**, select your repository, set main file path to `app.py`, and click **Deploy**.

---

## Key Data Science Insights

1. **Engineering Cohort Trends**: Students in Year 2 and Year 3 engineering coursework exhibit higher rates of anxiety and panic attacks compared to introductory years.
2. **Co-Occurrence of Conditions**: Students with reported anxiety show a strong positive correlation (`r = 0.27 - 0.53`) with depression.
3. **Specialist Treatment Gap**: Less than 10% of students reporting depressive symptoms had sought professional specialist treatment.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
