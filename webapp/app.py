import streamlit as st
import requests
import json
import os
import io
import time
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

st.set_page_config(page_title="Predictive Risk Pipeline", layout="wide")

# Helper: MinIO S3 client
@st.cache_resource
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def get_active_node_info():
    """Queries Kubernetes API for active worker pods in the airflow namespace and returns node placement info."""
    try:
        from kubernetes import client, config
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace="airflow")
        active_pods = []
        for p in pods.items:
            name = p.metadata.name
            phase = p.status.phase
            node = p.spec.node_name or "Scheduling..."
            if phase in ["Running", "Pending"] and not name.startswith(("airflow-postgres", "airflow-webserver", "airflow-scheduler")):
                active_pods.append(f"`{name}` -> Node: **{node}**")
        return active_pods
    except Exception:
        return []

# --- SIDEBAR: Execution History & Author ---
with st.sidebar:
    st.header("Predictive Risk Pipeline")
    st.markdown("**Author**: James Stephen  \n**Course**: CMPUT 441")
    st.markdown("---")
    
    st.subheader("Execution History")
    auto_refresh = st.toggle("Auto-Refresh (10s)", value=False)
    if st.button("Refresh History", use_container_width=True):
        st.rerun()

    try:
        resp = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns?limit=8&order_by=-execution_date",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=5
        )
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            if runs:
                for r in runs:
                    state = r.get("state", "unknown")
                    badge = "[SUCCESS]" if state == "success" else "[FAILED]" if state == "failed" else "[RUNNING]"
                    conf = r.get("conf", {})
                    ds = conf.get("dataset", "default")
                    th = conf.get("threshold", "0.50")
                    st.caption(f"{badge} **{r.get('dag_run_id')}**")
                    st.caption(f"Dataset: `{ds}` | Threshold: `{th}` | State: `{state.upper()}`")
                    st.markdown("---")
            else:
                st.info("No execution history yet.")
        else:
            st.error(f"Airflow API status: {resp.status_code}")
    except Exception as e:
        st.error(f"Airflow connection: {e}")

# --- MAIN TITLE & AUTHOR ATTRIBUTION ---
st.title("Predictive Risk Pipeline")
st.markdown("### Developed by James Stephen")
st.markdown("""
- **The Objective**: Evaluate the effectiveness of Support Vector Machines (SVM) with custom decision thresholds in detecting financial risk indicators.
- **The Use Case**: Determine whether client transactions are fraudulent or if a corporate loan applicant is heading toward bankruptcy. These classifiers provide vital risk mitigation before financial commitments are made.
""")

# --- PRESENTATION SLIDES SECTION ---
with st.expander("Presentation Deck (PDF)", expanded=False):
    pdf_path = os.path.join(os.path.dirname(__file__), "presentation.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Presentation PDF",
                data=f,
                file_name="Predictive_Risk_Pipeline_James_Stephen.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("Place `presentation.pdf` in the webapp directory to enable direct PDF downloads.")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("How the Kubernetes & Airflow Architecture Works", expanded=False):
        st.markdown("""
        - **Cloud-Native Execution**: Triggering a run sends an HTTP POST request to the internal **Airflow REST API**.
        - **Ephemeral Worker Pods**: Airflow dynamically schedules isolated `KubernetesPodOperator` pods onto x86_64 worker nodes (`node02`/`node03`).
        - **Distributed Object Storage**: Processed datasets, trained `.pkl` models, evaluation metrics, and charts are saved directly to S3-compatible **MinIO storage**.
        - **Decoupled Frontend**: Streamlit queries MinIO and Airflow to present real-time evaluation data without running heavy training locally.
        """)

with col_exp2:
    with st.expander("How the Machine Learning Model Works", expanded=False):
        st.markdown("""
        - **Data Preprocessing**: Splits data into an 80/20 train-test ratio and resamples the training set to a 1:1 class ratio to handle severe class imbalance without data leakage.
        - **Dimensionality Reduction**: Applies **L1-regularized Logistic Regression** for feature selection, successfully pruning 85 noisy features from the bankruptcy dataset to accelerate training.
        - **Model Construction**: Trains a **Support Vector Machine (SVM)** with an RBF kernel using 5-fold `GridSearchCV` (80 total fits across gamma and C parameters) to discover optimal decision margins.
        """)

st.markdown("---")

# --- UNIFIED CONTROLS & DYNAMIC DATASET CONTEXT ---
st.header("1. Pipeline Configuration & Control")

dataset_choice = st.selectbox(
    "Select Dataset", 
    options=["bankruptcy", "creditcard"],
    format_func=lambda x: "Company Bankruptcy Dataset" if x == "bankruptcy" else "Credit Card Fraud Dataset"
)

# Dynamic Dataset Context Box
if dataset_choice == "creditcard":
    st.info(
        "**Dataset Information (Credit Card Fraud)**: Sourced from Hugging Face, containing 284,807 transactions with 492 fraudulent cases (highly imbalanced). "
        "28 of the 30 features are anonymized numerical variables resulting from PCA. Target variable: `Class` (1 = Fraud, 0 = Normal)."
    )
else:
    st.info(
        "**Dataset Information (Company Bankruptcy)**: Sourced from UCI, containing 6,819 companies with 220 bankrupt cases. "
        "Contains 95 raw financial metrics such as interest ratios and debt ratios. Target variable: `Bankrupt?` (1 = Bankrupt, 0 = Solvent)."
    )

threshold_choice = st.slider(
    "Select Classification Threshold", 
    min_value=0.10, max_value=0.95, value=0.50, step=0.05,
    help="Custom acceptance threshold applied to SVM prediction probabilities."
)

# Dynamic Smart Threshold Guidance
if dataset_choice == "creditcard":
    st.caption(
        "💡 **Recommended Threshold: 0.80**. Maximizing recall with a 0.50 threshold results in 95% of fraud alerts being false positives, "
        "which severely degrades user trust. A 0.80 threshold maintains high recall (0.84) while boosting precision (0.16)."
    )
else:
    st.caption(
        "💡 **Recommended Threshold: 0.50**. In bankruptcy prediction, missing an impending bankruptcy (false negative) causes major loan defaults, "
        "whereas a false positive only requires manual financial auditing. Therefore, maintaining high recall at threshold 0.50 is optimal."
    )

s3 = get_s3_client()
prefix = f"{dataset_choice}/threshold_{threshold_choice}/"

# Check if model artifacts already exist in MinIO
artifacts_exist = False
try:
    s3.head_object(Bucket='models', Key=f"{prefix}classification_report.json")
    artifacts_exist = True
except Exception:
    artifacts_exist = False

run_btn = st.button("Run Pipeline on K3s Cluster", type="primary", use_container_width=True)
st.caption("ℹ️ **Cluster Parallelism**: You can trigger multiple pipeline runs in parallel. Airflow dynamically schedules worker pods onto worker nodes (`node02`/`node03`) while enforcing cluster resource quotas.")

if run_btn:
    endpoint = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    payload = {
        "conf": {
            "dataset": dataset_choice,
            "threshold": str(threshold_choice)
        }
    }
    
    with st.status("Initializing Airflow DAG Run...", expanded=True) as status_box:
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
                status_box.update(label=f"DAG Run Triggered (ID: {run_id}). Polling K3s worker pods...")
                
                start_time = time.time()
                completed = False
                
                while time.time() - start_time < 300:
                    time.sleep(3)
                    poll_resp = requests.get(
                        f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}",
                        auth=(AIRFLOW_USER, AIRFLOW_PASS),
                        timeout=5
                    )
                    if poll_resp.status_code == 200:
                        state = poll_resp.json().get("state", "queued")
                        
                        # Inspect node assignment from Kubernetes API
                        node_info = get_active_node_info()
                        if node_info:
                            node_str = " | Node Scheduling: " + ", ".join(node_info)
                        else:
                            node_str = ""
                            
                        status_box.update(label=f"Airflow Execution State: {state.upper()}{node_str}")
                        
                        if state == "success":
                            status_box.update(label="Pipeline Executed Successfully on K3s Cluster!", state="complete", expanded=False)
                            completed = True
                            break
                        elif state == "failed":
                            status_box.update(label="Pipeline Execution Failed", state="error", expanded=True)
                            st.error("DAG run failed. Check Airflow logs for details.")
                            break
                
                if completed:
                    st.rerun()
            else:
                status_box.update(label=f"Failed to trigger DAG: {response.status_code}", state="error")
                st.code(response.text)
        except Exception as e:
            status_box.update(label=f"Connection Error: {e}", state="error")

st.markdown("---")

# --- RESULTS & MODEL ARTIFACTS ---
st.header(f"2. Evaluation & Results ({dataset_choice.title()} | Threshold {threshold_choice})")

if artifacts_exist:
    try:
        report_obj = s3.get_object(Bucket='models', Key=f"{prefix}classification_report.json")
        report_data = json.loads(report_obj['Body'].read().decode('utf-8'))
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        accuracy = report_data.get('accuracy', 0.0)
        macro_avg = report_data.get('macro avg', {})
        
        m1.metric("Accuracy", f"{accuracy * 100:.1f}%")
        m2.metric("Macro Precision", f"{macro_avg.get('precision', 0.0):.2f}")
        m3.metric("Macro Recall", f"{macro_avg.get('recall', 0.0):.2f}")
        m4.metric("Macro F1-Score", f"{macro_avg.get('f1-score', 0.0):.2f}")

        # Classification Report Table
        st.subheader("Classification Report Breakdown")
        df_report = pd.DataFrame(report_data).transpose()
        st.dataframe(df_report.style.format("{:.3f}", na_rep="-"), use_container_width=True)

        # Visualizations Side-by-Side
        st.subheader("Evaluation Visualizations")
        img_col1, img_col2 = st.columns(2)
        
        try:
            cm_obj = s3.get_object(Bucket='models', Key=f"{prefix}confusion_matrix.png")
            cm_img = Image.open(io.BytesIO(cm_obj['Body'].read()))
            img_col1.image(cm_img, caption=f"Confusion Matrix ({dataset_choice} @ {threshold_choice})", use_container_width=True)
        except Exception:
            img_col1.warning("Confusion matrix chart not found.")

        try:
            roc_obj = s3.get_object(Bucket='models', Key=f"{prefix}roc_curve.png")
            roc_img = Image.open(io.BytesIO(roc_obj['Body'].read()))
            img_col2.image(roc_img, caption=f"ROC Curve ({dataset_choice})", use_container_width=True)
        except Exception:
            img_col2.warning("ROC curve chart not found.")

    except Exception as e:
        st.error(f"Error reading MinIO artifacts: {e}")
else:
    st.info(f"No trained model artifacts found for {dataset_choice} at threshold {threshold_choice}. Click 'Run Pipeline on K3s Cluster' above to generate them.")

if auto_refresh:
    time.sleep(10)
    st.rerun()