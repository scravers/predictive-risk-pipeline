import streamlit as st
import requests
import json
import os
import io
import boto3
import pandas as pd
from PIL import Image

# --- Configuration ---
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver.airflow.svc.cluster.local:8080")
DAG_ID = "financial_risk_pipeline"

AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "adminpassword")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")

st.set_page_config(page_title="Financial Risk ML Pipeline", page_icon="🏦", layout="wide")

# Helper: MinIO S3 client
@st.cache_resource
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

st.title("🏦 Financial Risk ML Pipeline Dashboard")
st.markdown("Trigger distributed K3s machine learning workflows and explore trained models, evaluation metrics, and charts from MinIO.")

tabs = st.tabs(["🚀 Trigger Pipeline", "📊 Model Artifacts & Evaluation", "🔄 Active DAG Runs"])

# --- TAB 1: TRIGGER PIPELINE ---
with tabs[0]:
    st.header("1. Pipeline Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_choice = st.selectbox(
            "Select Dataset", 
            options=["bankruptcy", "creditcard"],
            format_func=lambda x: "Company Bankruptcy Dataset" if x == "bankruptcy" else "Credit Card Fraud Dataset"
        )
    
    with col2:
        threshold_choice = st.slider(
            "Select Classification Threshold", 
            min_value=0.10, max_value=0.95, value=0.50, step=0.05,
            help="Higher thresholds increase precision (fewer false positives) but lower recall."
        )

    st.markdown("---")
    if st.button("🚀 Run Pipeline on K3s", type="primary", use_container_width=True):
        endpoint = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
        payload = {
            "conf": {
                "dataset": dataset_choice,
                "threshold": str(threshold_choice)
            }
        }
        
        with st.spinner("Communicating with Airflow API..."):
            try:
                response = requests.post(
                    endpoint,
                    auth=(AIRFLOW_USER, AIRFLOW_PASS),
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=10
                )
                
                if response.status_code == 200:
                    run_id = response.json().get("dag_run_id")
                    st.success(f"✅ Pipeline successfully triggered! (Run ID: `{run_id}`)")
                    st.info("Pods are running on your worker nodes. Switch to the **Model Artifacts** tab once training completes.")
                else:
                    st.error(f"❌ Failed to trigger pipeline. Status code: {response.status_code}")
                    st.code(response.text)
                    
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

# --- TAB 2: MODEL ARTIFACTS & EVALUATION ---
with tabs[1]:
    st.header("📊 Trained Models & Evaluation Metrics")
    s3 = get_s3_client()
    
    col_ds, col_th = st.columns(2)
    with col_ds:
        sel_dataset = st.selectbox("Select Dataset View", ["bankruptcy", "creditcard"], key="view_dataset")
    with col_th:
        sel_threshold = st.slider("Select Threshold View", 0.10, 0.95, 0.50, step=0.05, key="view_threshold")

    prefix = f"{sel_dataset}/threshold_{sel_threshold}/"
    st.caption(f"MinIO Path: `s3://models/{prefix}`")
    
    try:
        # Check if classification report exists in S3
        report_obj = s3.get_object(Bucket='models', Key=f"{prefix}classification_report.json")
        report_data = json.loads(report_obj['Body'].read().decode('utf-8'))
        
        # Display Key Metrics Cards
        st.subheader("Classification Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        accuracy = report_data.get('accuracy', 0.0)
        macro_avg = report_data.get('macro avg', {})
        weighted_avg = report_data.get('weighted avg', {})
        
        m1.metric("Accuracy", f"{accuracy * 100:.1f}%")
        m2.metric("Macro Precision", f"{macro_avg.get('precision', 0.0):.2f}")
        m3.metric("Macro Recall", f"{macro_avg.get('recall', 0.0):.2f}")
        m4.metric("Macro F1-Score", f"{macro_avg.get('f1-score', 0.0):.2f}")

        # Classification Report Table
        st.subheader("Detailed Classification Report")
        df_report = pd.DataFrame(report_data).transpose()
        st.dataframe(df_report.style.format("{:.3f}", na_rep="-"), use_container_width=True)

        # Charts Display
        st.subheader("Visualizations")
        img_col1, img_col2 = st.columns(2)
        
        try:
            cm_obj = s3.get_object(Bucket='models', Key=f"{prefix}confusion_matrix.png")
            cm_img = Image.open(io.BytesIO(cm_obj['Body'].read()))
            img_col1.image(cm_img, caption=f"Confusion Matrix ({sel_dataset} @ {sel_threshold})", use_container_width=True)
        except Exception:
            img_col1.warning("Confusion matrix chart not found.")

        try:
            roc_obj = s3.get_object(Bucket='models', Key=f"{prefix}roc_curve.png")
            roc_img = Image.open(io.BytesIO(roc_obj['Body'].read()))
            img_col2.image(roc_img, caption=f"ROC Curve ({sel_dataset} @ {sel_threshold})", use_container_width=True)
        except Exception:
            img_col2.warning("ROC curve chart not found.")

    except s3.exceptions.NoSuchKey:
        st.warning(f"No completed run found for dataset `{sel_dataset}` at threshold `{sel_threshold}`. Trigger a run in Tab 1!")
    except Exception as e:
        st.error(f"Could not connect to MinIO or fetch artifacts: {e}")

# --- TAB 3: ACTIVE DAG RUNS ---
with tabs[2]:
    st.header("🔄 Airflow DAG Execution History")
    if st.button("🔄 Refresh Status"):
        st.rerun()
        
    try:
        resp = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns?limit=10&order_by=-execution_date",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=5
        )
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            if runs:
                df_runs = pd.DataFrame([{
                    "Run ID": r.get("dag_run_id"),
                    "State": r.get("state"),
                    "Execution Date": r.get("execution_date"),
                    "Dataset": r.get("conf", {}).get("dataset", "default"),
                    "Threshold": r.get("conf", {}).get("threshold", "default")
                } for r in runs])
                st.dataframe(df_runs, use_container_width=True)
            else:
                st.info("No DAG runs found yet.")
        else:
            st.error(f"Failed to fetch DAG runs: {resp.status_code}")
    except Exception as e:
        st.error(f"Could not connect to Airflow: {e}")