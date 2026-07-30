import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }
        .stHeader {
            background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
            padding: 1.8rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            margin-bottom: 1.5rem;
        }
        .header-title {
            color: #f8fafc;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
        }
        .badge-fraud {
            background-color: #ef4444;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.4rem;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        }
        .badge-safe {
            background-color: #22c55e;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.4rem;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);
        }
        .metric-box {
            background: #1e293b;
            border: 1px solid #334155;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_model_and_scaler()

NORMAL_SAMPLE = {
    'Time': 40000.0, 'Amount': 45.50,
    'V1': -0.4, 'V2': 0.1, 'V3': 1.2, 'V4': -0.5, 'V5': 0.3,
    'V6': -0.2, 'V7': 0.4, 'V8': 0.05, 'V9': 0.2, 'V10': 0.1,
    'V11': -0.3, 'V12': 0.2, 'V13': -0.1, 'V14': 0.3, 'V15': -0.2,
    'V16': 0.1, 'V17': 0.2, 'V18': -0.1, 'V19': 0.05, 'V20': -0.05,
    'V21': -0.02, 'V22': 0.01, 'V23': -0.01, 'V24': 0.02, 'V25': 0.01,
    'V26': -0.01, 'V27': 0.01, 'V28': 0.01
}

FRAUD_SAMPLE = {
    'Time': 75000.0, 'Amount': 250.00,
    'V1': -4.5, 'V2': 4.2, 'V3': -7.5, 'V4': 5.8, 'V5': -4.2,
    'V6': -1.8, 'V7': -6.5, 'V8': 2.1, 'V9': -3.8, 'V10': -8.2,
    'V11': 6.1, 'V12': -9.4, 'V13': -0.5, 'V14': -10.5, 'V15': 0.4,
    'V16': -6.2, 'V17': -12.1, 'V18': -4.8, 'V19': 2.5, 'V20': 0.8,
    'V21': 1.2, 'V22': -0.4, 'V23': -0.5, 'V24': 0.1, 'V25': 0.4,
    'V26': 0.2, 'V27': 0.8, 'V28': 0.3
}

st.sidebar.title("Credit Card Fraud AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Mode",
    ["Single Transaction Predictor", "Batch CSV Prediction", "Dataset Insights & Metrics"]
)

st.sidebar.markdown("---")
st.sidebar.info("Model: Regularized XGBoost Classifier\nScaler: RobustScaler\nAUPRC: 0.8636 | Recall: 85.71%")

st.markdown("""
    <div class="stHeader">
        <div class="header-title">Credit Card Fraud Detection Portal</div>
        <div class="header-subtitle">Real-time Fraud Prediction System powered by Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

if page == "Single Transaction Predictor":
    st.subheader("Single Transaction Fraud Predictor")
    st.markdown("Test individual transaction parameters or use instant pre-fill sample buttons below.")
    
    col_b1, col_b2 = st.columns(2)
    
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = NORMAL_SAMPLE.copy()
        
    with col_b1:
        if st.button("Load Normal Sample Transaction", use_container_width=True):
            st.session_state['input_data'] = NORMAL_SAMPLE.copy()
            st.rerun()
            
    with col_b2:
        if st.button("Load Fraud Sample Transaction", use_container_width=True):
            st.session_state['input_data'] = FRAUD_SAMPLE.copy()
            st.rerun()

    current_inputs = st.session_state['input_data']

    st.markdown("---")
    with st.form("single_pred_form"):
        st.markdown("#### Transaction Overview")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", value=float(current_inputs['Amount']), min_value=0.0, step=5.0)
        with c2:
            time_val = st.number_input("Elapsed Time (Seconds)", value=float(current_inputs['Time']), min_value=0.0, step=100.0)
            
        st.markdown("#### Key Risk Features (PCA Components)")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            v14 = st.number_input("V14 (Strong Neg Corr)", value=float(current_inputs['V14']))
            v17 = st.number_input("V17 (Strong Neg Corr)", value=float(current_inputs['V17']))
        with c_p2:
            v12 = st.number_input("V12 (Neg Corr)", value=float(current_inputs['V12']))
            v10 = st.number_input("V10 (Neg Corr)", value=float(current_inputs['V10']))
        with c_p3:
            v11 = st.number_input("V11 (Strong Pos Corr)", value=float(current_inputs['V11']))
            v4 = st.number_input("V4 (Pos Corr)", value=float(current_inputs['V4']))
        with c_p4:
            v2 = st.number_input("V2 (Pos Corr)", value=float(current_inputs['V2']))
            v19 = st.number_input("V19 (Pos Corr)", value=float(current_inputs['V19']))

        with st.expander("All PCA Components (V1 to V28)"):
            adv_cols = st.columns(4)
            pca_inputs = {}
            for i in range(1, 29):
                feat_name = f'V{i}'
                col_idx = (i - 1) % 4
                with adv_cols[col_idx]:
                    val = float(current_inputs.get(feat_name, 0.0))
                    pca_inputs[feat_name] = st.number_input(feat_name, value=val, key=f"input_{feat_name}")

        submit_btn = st.form_submit_button("Predict Fraud Risk", use_container_width=True)

    if submit_btn:
        if model is None or scaler is None:
            st.error("Model or scaler files not found in models/ directory.")
        else:
            pca_inputs['V14'] = v14
            pca_inputs['V17'] = v17
            pca_inputs['V12'] = v12
            pca_inputs['V10'] = v10
            pca_inputs['V11'] = v11
            pca_inputs['V4'] = v4
            pca_inputs['V2'] = v2
            pca_inputs['V19'] = v19
            
            scaled_amt = scaler.transform(np.array([[amount]]))[0][0]
            scaled_tm = scaler.transform(np.array([[time_val]]))[0][0]
            
            feature_order = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
            input_dict = {**pca_inputs, 'scaled_amount': scaled_amt, 'scaled_time': scaled_tm}
            input_df = pd.DataFrame([input_dict])[feature_order]
            
            probability = model.predict_proba(input_df)[0][1]
            is_fraud = probability >= 0.5
            
            st.markdown("---")
            st.subheader("Prediction Result & Risk Assessment")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                if is_fraud:
                    st.markdown("""
                        <div class="badge-fraud">
                            FRAUD DETECTED<br>
                            <span style="font-size:0.9rem; font-weight:normal;">Action Required: Block Transaction</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="badge-safe">
                            LEGITIMATE TRANSACTION<br>
                            <span style="font-size:0.9rem; font-weight:normal;">Approved</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with r_col2:
                st.markdown(f"**Calculated Fraud Risk Score:** `{probability * 100:.2f}%`")
                st.progress(float(probability))
                
                if is_fraud:
                    st.warning("High Risk Alert: Highly anomalous feature values detected (e.g. V14, V17). Recommend immediate card freeze.")
                else:
                    st.success("Transaction approved. Features fall within legitimate historical patterns.")

elif page == "Batch CSV Prediction":
    st.subheader("Batch CSV Fraud Prediction")
    st.markdown("Upload a CSV file containing transactions to run automated batch predictions.")
    
    sample_df = pd.DataFrame([NORMAL_SAMPLE, FRAUD_SAMPLE])
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Sample CSV Template for Testing",
        data=sample_csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(batch_data):,} transactions.")
            
            if st.button("Run Batch Prediction", use_container_width=True):
                req_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                missing = [c for c in req_cols if c not in batch_data.columns]
                
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    proc_df = batch_data.copy()
                    proc_df['scaled_amount'] = scaler.transform(proc_df['Amount'].values.reshape(-1, 1))
                    proc_df['scaled_time'] = scaler.transform(proc_df['Time'].values.reshape(-1, 1))
                    
                    feature_order = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
                    X_batch = proc_df[feature_order]
                    
                    probs = model.predict_proba(X_batch)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    batch_data['Fraud_Probability (%)'] = np.round(probs * 100, 2)
                    batch_data['Prediction'] = np.where(preds == 1, 'FRAUD', 'LEGITIMATE')
                    
                    fraud_cnt = (preds == 1).sum()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Records", f"{len(batch_data):,}")
                    m2.metric("Flagged Frauds", f"{fraud_cnt:,}")
                    m3.metric("Fraud Percentage", f"{(fraud_cnt/len(batch_data))*100:.2f}%")
                    
                    st.dataframe(batch_data[['Time', 'Amount', 'Fraud_Probability (%)', 'Prediction'] + [f'V{i}' for i in range(1, 5)]], use_container_width=True)
                    
                    output_csv = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predicted Predictions CSV",
                        data=output_csv,
                        file_name="scored_fraud_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

elif page == "Dataset Insights & Metrics":
    st.subheader("Model Performance & Dataset Insights")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Size", "284,807")
    m2.metric("Total Frauds", "492")
    m3.metric("Precision", "76.36%")
    m4.metric("Recall", "85.71%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("notebook/images/class_distribution.png"):
            st.image("notebook/images/class_distribution.png", caption="Target Class Imbalance")
    with col2:
        if os.path.exists("notebook/images/overfitting_comparison.png"):
            st.image("notebook/images/overfitting_comparison.png", caption="Overfitting Diagnostics (Train vs Test AUPRC)")
