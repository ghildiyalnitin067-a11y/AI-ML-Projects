import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(
    page_title="Salary Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Employee Salary Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict estimated annual salary based on age, education level, job title, and years of experience.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    data_path = os.path.join(base_dir, 'dataset', 'Salary Data.csv')
    metrics_path = os.path.join(base_dir, 'models', 'metrics.json')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please train the model first.")
        st.stop()
        
    model = joblib.load(model_path)
    
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
            
    return model, df, metrics

model, df, metrics = load_assets()

st.sidebar.header("Model Metrics")
if metrics:
    st.sidebar.info(f"**Model:** {metrics.get('model_name', 'Tuned XGBoost')}")
    st.sidebar.metric("Test R2 Score", f"{metrics.get('test_r2', 0.8872):.2%}")
    st.sidebar.metric("RMSE (Avg Error)", f"${metrics.get('rmse', 14597.43):,.2f}")
    st.sidebar.metric("5-Fold CV Score", f"{metrics.get('cv_mean_r2', 0.8852):.2%}")
else:
    st.sidebar.write("Model: Tuned XGBoost")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Scikit-Learn, XGBoost & Streamlit")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic & Education Info")
    
    age = st.slider("Select Age (Years)", min_value=21, max_value=65, value=30, step=1)
    
    genders = sorted(df['Gender'].unique().tolist())
    gender = st.selectbox("Select Gender", options=genders)
    
    education_levels = sorted(df['Education Level'].unique().tolist())
    education_level = st.selectbox("Select Education Level", options=education_levels)

with col2:
    st.subheader("Professional Experience")
    
    job_titles = sorted(df['Job Title'].unique().tolist())
    default_job_idx = job_titles.index('Software Engineer') if 'Software Engineer' in job_titles else 0
    job_title = st.selectbox("Select Job Title", options=job_titles, index=default_job_idx)
    
    years_exp = st.slider("Years of Experience", min_value=0.0, max_value=35.0, value=5.0, step=0.5)

st.markdown("---")

center_col = st.columns([1, 2, 1])[1]

with center_col:
    predict_btn = st.button("Estimate Salary", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_exp
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        
        st.success("### Prediction Complete!")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Estimated Salary", f"${prediction:,.2f}")
        res_col2.metric("Monthly Salary", f"${(prediction / 12):,.2f}")
        res_col3.metric("Years of Exp", f"{years_exp} Yrs")
        
        st.info(f"**Insight:** For a **{job_title}** with a **{education_level}** degree and **{years_exp} years** of experience, the predicted baseline compensation is **${prediction:,.2f}** per year.")
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Salary Prediction ML App • Built for Internship & Portfolio Demonstration</p>", unsafe_allow_html=True)
