import os
import joblib
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

def train_and_optimize_models(cleaned_csv_path: str, course_filter: str = None):
    """Load cleaned data, optionally filter by course, split, tune hyperparameters for core & advanced ML algorithms, evaluate, and save the best model."""
    if not os.path.exists(cleaned_csv_path):
        from src.data_preprocessing import load_data, clean_data, save_cleaned_data
        raw_path = os.path.join('dataset', 'Student Mental health.csv')
        df = clean_data(load_data(raw_path))
        save_cleaned_data(df, cleaned_csv_path)
    else:
        df = pd.read_csv(cleaned_csv_path)

    if course_filter:
        df = df[df['course'] == course_filter].reset_index(drop=True)
        print(f"Filtered dataset for course '{course_filter}' (Total records: {len(df)})")

    # Features and Target selection
    feature_cols = [
        'age', 'gender_binary', 'year_of_study_num', 'cgpa_ordinal', 
        'marital_status', 'anxiety', 'panic_attack'
    ]
    target_col = 'depression'

    X = df[feature_cols]
    y = df[target_col]

    # Train-Test Split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Comprehensive ML Algorithm Suite & Hyperparameter Grids
    models_config = {
        'XGBoost Classifier': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'Random Forest Classifier': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [25, 50, 100],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 4]
            }
        },
        'Decision Tree Classifier': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [2, 3, 5, 7, None],
                'criterion': ['gini', 'entropy']
            }
        },
        'Gradient Boosting (GBM)': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 5]
            }
        },
        'Stochastic Gradient Descent (SGD)': {
            'model': SGDClassifier(random_state=42, max_iter=2000),
            'params': {
                'loss': ['log_loss', 'hinge', 'modified_huber'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            }
        },
        'Logistic Regression (L1/L2)': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Support Vector Machine (SVM)': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        }
    }

    results = []
    best_overall_score = -1
    best_overall_model = None
    best_overall_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=======================================================")
    print("      MODEL TRAINING & HYPERPARAMETER OPTIMIZATION     ")
    print("=======================================================\n")

    for name, config in models_config.items():
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            'Model': name,
            'Best Parameters': str(grid.best_params_),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        print(f"-> Model: {name}")
        print(f"   Best Params: {grid.best_params_}")
        print(f"   Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}\n")

        if f1 > best_overall_score or (f1 == best_overall_score and acc > (results[0]['Accuracy'] if results else 0)):
            best_overall_score = f1
            best_overall_model = best_model
            best_overall_name = name

    results_df = pd.DataFrame(results)
    
    # Save model artifacts using absolute BASE_DIR
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    joblib.dump(best_overall_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[BEST MODEL SELECTED]: {best_overall_name} (Saved to {model_path})")
    return results_df, best_overall_model, scaler

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(BASE_DIR, 'dataset', 'cleaned_student_mental_health.csv')
    train_and_optimize_models(cleaned_path)
