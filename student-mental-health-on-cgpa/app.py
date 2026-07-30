import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.data_preprocessing import load_data, clean_data
from src.database import check_mongo_connection, log_prediction, fetch_predictions_history

# Base Directory Resolution for Cross-Platform Cloud Deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Student Mental Health ML Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load & Cache Clean Data
@st.cache_data
def get_clean_dataset():
    raw_path = os.path.join(BASE_DIR, 'dataset', 'Student Mental health.csv')
    cleaned_path = os.path.join(BASE_DIR, 'dataset', 'cleaned_student_mental_health.csv')
    if os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path)
    df_clean = clean_data(load_data(raw_path))
    df_clean.to_csv(cleaned_path, index=False)
    return df_clean

# Load & Cache Model Artifacts — load once per app lifecycle
@st.cache_resource(show_spinner="Loading ML model...")
def get_model_artifacts():
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        # Fallback: only triggered if pkl files are missing (should not happen on cloud)
        from src.train_model import train_and_optimize_models
        cleaned_csv_path = os.path.join(BASE_DIR, 'dataset', 'cleaned_student_mental_health.csv')
        train_and_optimize_models(cleaned_csv_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Cache MongoDB connection check — runs ONCE per session, not on every page render
@st.cache_resource(show_spinner=False)
def get_mongo_status():
    return check_mongo_connection()

df = get_clean_dataset()

# Sidebar Navigation & Filters
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", ["Exploratory Data Analysis", "Model Performance", "Live Depression Risk Predictor", "MongoDB Prediction History"])

st.sidebar.markdown("---")
st.sidebar.title("Cohort Filter")
cohort_option = st.sidebar.selectbox("Filter Students by Major", ["All Students", "Engineering Cohort Only"])


filtered_df = df.copy()
if cohort_option == "Engineering Cohort Only":
    filtered_df = filtered_df[filtered_df['course'] == 'engineering'].reset_index(drop=True)

# MongoDB Connection — cached, runs once per session
st.sidebar.markdown("---")
st.sidebar.title("Database Status")
is_mongo_connected, mongo_msg = get_mongo_status()

if is_mongo_connected:
    st.sidebar.success("MongoDB Status: Connected")
    st.sidebar.caption(mongo_msg)
else:
    st.sidebar.warning("MongoDB Status: Not Connected (Optional)")
    st.sidebar.caption("Predictions will run locally. Set MONGO_URI in .env or Streamlit Secrets to enable logging.")

# Page 1: Exploratory Data Analysis
if page == "Exploratory Data Analysis":
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown('<div class="main-title">Student Mental Health & CGPA Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Exploratory Data Analysis for <b>{cohort_option}</b> (Total Records: {len(filtered_df)})</div>', unsafe_allow_html=True)

    # Key Summary Indicators
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(filtered_df))
    col2.metric("Depression Rate", f"{filtered_df['depression'].mean() * 100:.1f}%")
    col3.metric("Anxiety Rate", f"{filtered_df['anxiety'].mean() * 100:.1f}%")
    col4.metric("Panic Attack Rate", f"{filtered_df['panic_attack'].mean() * 100:.1f}%")

    st.markdown("---")

    # Chart Section 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CGPA Band Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df,
            x='cgpa',
            hue='cgpa',
            order=['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00'],
            palette='Blues_d',
            legend=False,
            ax=ax1
        )
        ax1.set_xlabel("CGPA Range")
        ax1.set_ylabel("Student Count")
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        st.subheader("Gender Breakdown")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='gender', hue='gender', palette='Set2', legend=False, ax=ax2)
        ax2.set_xlabel("Gender")
        ax2.set_ylabel("Student Count")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

    # Chart Section 2: Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    corr_cols = ['age', 'year_of_study_num', 'cgpa_ordinal', 'marital_status', 'depression', 'anxiety', 'panic_attack']
    sns.heatmap(filtered_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, vmin=-1, vmax=1)
    st.pyplot(fig3)
    plt.close(fig3)

# Page 2: Model Performance
elif page == "Model Performance":
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown('<div class="main-title">Machine Learning Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hyperparameter Optimization with GridSearchCV & 5-Fold Stratified Cross-Validation</div>', unsafe_allow_html=True)

    model_data = [
        {"Model": "Random Forest Classifier", "Accuracy": 0.8462, "Precision": 1.0000, "Recall": 0.5556, "F1-Score": 0.7143, "Best Hyperparameters": "{'max_depth': 5, 'n_estimators': 25}"},
        {"Model": "Decision Tree Classifier", "Accuracy": 0.8077, "Precision": 0.7500, "Recall": 0.6667, "F1-Score": 0.7059, "Best Hyperparameters": "{'criterion': 'entropy', 'max_depth': 7}"},
        {"Model": "Gradient Boosting (GBM)", "Accuracy": 0.7692, "Precision": 0.7143, "Recall": 0.5556, "F1-Score": 0.6250, "Best Hyperparameters": "{'learning_rate': 0.1, 'max_depth': 3}"},
        {"Model": "Support Vector Machine (SVM)", "Accuracy": 0.7692, "Precision": 0.7143, "Recall": 0.5556, "F1-Score": 0.6250, "Best Hyperparameters": "{'C': 10.0, 'kernel': 'rbf'}"},
        {"Model": "XGBoost Classifier", "Accuracy": 0.8077, "Precision": 1.0000, "Recall": 0.4444, "F1-Score": 0.6154, "Best Hyperparameters": "{'learning_rate': 0.05, 'max_depth': 3}"},
        {"Model": "Logistic Regression (L1/L2)", "Accuracy": 0.8077, "Precision": 1.0000, "Recall": 0.4444, "F1-Score": 0.6154, "Best Hyperparameters": "{'C': 1.0, 'solver': 'lbfgs'}"},
        {"Model": "Stochastic Gradient Descent (SGD)", "Accuracy": 0.7692, "Precision": 0.8000, "Recall": 0.4444, "F1-Score": 0.5714, "Best Hyperparameters": "{'alpha': 0.001, 'penalty': 'elasticnet'}"}
    ]

    perf_df = pd.DataFrame(model_data)
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=perf_df, x='Model', y='F1-Score', palette='mako', ax=ax4)
    plt.xticks(rotation=20)
    ax4.set_ylim(0, 1.0)
    st.pyplot(fig4)
    plt.close(fig4)

# Page 3: Live Depression Risk Predictor
elif page == "Live Depression Risk Predictor":
    st.markdown('<div class="main-title">Live Student Depression Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter student demographic and academic performance details below to evaluate mental health risk probability.</div>', unsafe_allow_html=True)

    model, scaler = get_model_artifacts()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age_val = st.slider("Student Age", min_value=17, max_value=30, value=20)
            gender_str = st.selectbox("Gender", ["Female", "Male"])
            year_val = st.slider("Year of Study", min_value=1, max_value=4, value=2)

        with col2:
            cgpa_str = st.selectbox("CGPA Range", ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"], index=3)
            marital_str = st.selectbox("Marital Status", ["No", "Yes"])
            anxiety_str = st.selectbox("Do you experience Anxiety?", ["No", "Yes"])
            panic_str = st.selectbox("Do you experience Panic Attacks?", ["No", "Yes"])

        submit_btn = st.form_submit_button("Predict Mental Health Risk")

    if submit_btn:
        # Map input features
        gender_bin = 1 if gender_str == "Male" else 0
        cgpa_map = {"0 - 1.99": 1, "2.00 - 2.49": 2, "2.50 - 2.99": 3, "3.00 - 3.49": 4, "3.50 - 4.00": 5}
        cgpa_ord = cgpa_map[cgpa_str]
        marital_bin = 1 if marital_str == "Yes" else 0
        anxiety_bin = 1 if anxiety_str == "Yes" else 0
        panic_bin = 1 if panic_str == "Yes" else 0

        # Feature vector: ['age', 'gender_binary', 'year_of_study_num', 'cgpa_ordinal', 'marital_status', 'anxiety', 'panic_attack']
        input_features = np.array([[age_val, gender_bin, year_val, cgpa_ord, marital_bin, anxiety_bin, panic_bin]])
        input_scaled = scaler.transform(input_features)

        pred_class = int(model.predict(input_scaled)[0])
        pred_proba = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, 'predict_proba') else None

        st.markdown("---")
        if pred_class == 1 or (pred_proba is not None and pred_proba >= 0.5):
            st.error(f"High Risk of Depression Detected ({pred_proba * 100:.1f}% Estimated Probability)" if pred_proba else "High Risk of Depression Detected")
            st.info("Recommendation: Consider consulting an academic counselor or specialist for mental health support.")
        else:
            st.success(f"Low Risk of Depression Detected ({(1 - pred_proba) * 100:.1f}% Confidence)" if pred_proba else "Low Risk of Depression Detected")
            st.info("Student parameters indicate a stable risk profile.")

        # Log prediction to MongoDB if database is connected
        record_payload = {
            "age": age_val,
            "gender": gender_str,
            "year_of_study": year_val,
            "cgpa_range": cgpa_str,
            "marital_status": marital_str,
            "anxiety": anxiety_str,
            "panic_attack": panic_str,
            "predicted_class": pred_class,
            "risk_level": "High Risk" if pred_class == 1 else "Low Risk",
            "probability_percent": round(pred_proba * 100, 2) if pred_proba is not None else None
        }

        db_success, db_msg = log_prediction(record_payload)
        if db_success:
            st.success(f"Prediction successfully saved to MongoDB (ID: {db_msg})")
        else:
            st.caption(f"Database Notice: {db_msg}")

# Page 4: MongoDB Prediction History
elif page == "MongoDB Prediction History":
    st.markdown('<div class="main-title">MongoDB Prediction Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time database logs saved from user prediction submissions.</div>', unsafe_allow_html=True)

    if not is_mongo_connected:
        st.warning("MongoDB connection is currently offline. Please configure MONGO_URI in .env file to enable history view.")
    else:
        history = fetch_predictions_history(limit=100)
        if not history:
            st.info("No predictions recorded yet in MongoDB. Try submitting a prediction in the Predictor tab!")
        else:
            st.write(f"Total Database Logs Fetched: **{len(history)}**")
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)
