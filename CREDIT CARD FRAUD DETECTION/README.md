# Credit Card Fraud Detection AI System

An end-to-end Machine Learning and Real-Time Web Platform for detecting fraudulent credit card transactions. Powered by Regularized XGBoost, SMOTE resampling, Stratified 5-Fold Cross-Validation, and Streamlit.

---

## Live Streamlit Application

Streamlit App URL: 

(Paste your deployed Streamlit URL above)

---

## Key Highlights & Performance

- **Target Class Imbalance**: Handled extreme 0.17% fraud ratio (492 frauds vs 284,315 normal transactions).
- **Overfitting Diagnostics**: Mitigated model memorization using Stratified 5-Fold Cross-Validation with `imblearn.pipeline.Pipeline`.
- **Top Metrics**:
  - **AUPRC (Precision-Recall AUC)**: `0.8636`
  - **Recall (Fraud Catch Rate)**: `85.71%` (84 out of 98 test fraud cases caught)
  - **Precision**: `76.36%` (Reduced false positive blocks from 152 to only 26)
  - **F1-Score**: `0.8077`
  - **ROC-AUC**: `0.9775`

---

## Project Features

1. **Single Transaction Risk Predictor**:
   - Instant single-transaction risk scoring.
   - Includes one-click test sample presets (`Load Normal Sample`, `Load Fraud Sample`).
   - Visual risk score percentage bar & action recommendations.

2. **Batch CSV Prediction**:
   - Upload transaction `.csv` files for automated batch risk scoring.
   - Provides sample downloadable CSV template.
   - Generates downloadable prediction results CSV.

3. **Exploratory Data Analysis (EDA)**:
   - Interactive visual insights covering class distribution, transaction amount analysis, hourly density, and PCA feature correlation matrix.

4. **Notebook Workflows**:
   - `notebook/EDA.ipynb`: Full exploratory data analysis.
   - `notebook/Model_Training.ipynb`: Preprocessing, SMOTE, 5-Fold CV, hyperparameter tuning, and regularization checks.

---

## Project Structure

```
CREDIT CARD FRAUD DETECTION/
├── app.py                      # Streamlit Web Application
├── models/
│   ├── best_model.pkl          # Trained Regularized XGBoost Classifier
│   └── scaler.pkl              # Fitted RobustScaler
├── notebook/
│   ├── EDA.ipynb               # Exploratory Data Analysis Notebook
│   ├── Model_Training.ipynb    # Model Training & Optimization Notebook
│   └── images/                 # Saved Visualization PNGs
├── data/
│   └── creditcard.csv          # Raw Dataset (Ignored in Git)
├── requirement.txt             # Python Dependencies
├── .gitignore                  # Excluded Files & Directories
└── README.md                   # Project Documentation
```

---

## Installation & Local Execution

### 1. Clone Repository
```bash
git clone https://github.com/ghildiyalnitin067-a11y/AI-ML-Projects.git
cd "AI-ML-Projects/CREDIT CARD FRAUD DETECTION"
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```
