import streamlit as st
import requests
import json
import os
import io
import time
import boto3
import pandas as pd
import kubernetes
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

def get_active_dag_runs():
    """Fetches active (running or queued) DAG runs from Airflow."""
    try:
        resp = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns?limit=10&order_by=-execution_date",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=4
        )
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            return [r for r in runs if r.get("state") in ["running", "queued"]]
    except Exception:
        pass
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
                    run_id = r.get("dag_run_id")
                    st.caption(f"{badge} **{run_id}**")
                    st.caption(f"Dataset: `{ds}` | Threshold: `{th}` | State: `{state.upper()}`")
                    st.markdown("---")
            else:
                st.info("No execution history yet.")
        else:
            st.error(f"Airflow API status: {resp.status_code}")
    except Exception as e:
        st.error(f"Airflow connection: {e}")

# --- HEADER & PROJECT OVERVIEW ---
st.title("Predictive Risk Pipeline")
st.markdown("### Developed by James Stephen")
st.markdown("""
- **Objective**: Evaluate Support Vector Machines (SVM) with custom decision thresholds for financial risk detection.
- **Use Case**: Identify credit card fraud transactions and corporate bankruptcy risks before financial commitments are made.
""")

col_hdr1, col_hdr2 = st.columns([3, 1])

with col_hdr1:
    with st.expander("System Architecture & ML Model Overview", expanded=False):
        st.markdown("""
        **Infrastructure & Cloud-Native Execution**:
        - **K3s Bare-Metal Cluster**: Airflow dynamically schedules worker pods onto x86_64 worker nodes (`node02`/`node03`).
        - **Decoupled Architecture**: Frontend (Streamlit), Orchestration (Airflow REST API), and Storage (MinIO S3) run independently in isolated namespaces.

        **Machine Learning Pipeline**:
        - **Preprocessing**: 80/20 train-test split with 1:1 downsampling to eliminate severe class imbalance without data leakage.
        - **Feature Selection**: L1-regularized Logistic Regression prunes 85 noisy features from the bankruptcy dataset.
        - **SVM Training**: RBF-kernel SVM tuned via 5-fold `GridSearchCV` (80 fits) to discover optimal decision boundaries.
        """)

with col_hdr2:
    pdf_path = os.path.join(os.path.dirname(__file__), "presentation.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Presentation PDF",
                data=f,
                file_name="Predictive_Risk_Pipeline_James_Stephen.pdf",
                mime="application/pdf",
                use_container_width=True
            )

st.markdown("---")

# --- PIPELINE CONTROLS ---
st.header("1. Pipeline Configuration & Control")

def format_dataset_label(x):
    if x == "bankruptcy":
        return "Company Bankruptcy Dataset (UCI | 6,819 instances, 95 features)"
    elif x == "creditcard":
        return "Credit Card Fraud Dataset (Hugging Face | 284,807 instances, 30 features)"
    return str(x)

dataset_choice = st.selectbox(
    "Select Dataset", 
    options=["bankruptcy", "creditcard"],
    format_func=format_dataset_label
)

if dataset_choice == "creditcard":
    st.info(
        "**Dataset Information (Credit Card Fraud)**: Sourced from Hugging Face, containing 284,807 transactions with 492 fraudulent cases. "
        "28 of 30 features are PCA components. Target variable: `Class` (1 = Fraud, 0 = Normal)."
    )
else:
    st.info(
        "**Dataset Information (Company Bankruptcy)**: Sourced from UCI, containing 6,819 companies with 220 bankrupt cases. "
        "Contains 95 raw financial metrics. Target variable: `Bankrupt?` (1 = Bankrupt, 0 = Solvent)."
    )

threshold_choice = st.slider(
    "Select Classification Threshold", 
    min_value=0.10, max_value=0.95, value=0.50, step=0.05,
    help="Custom acceptance threshold applied to SVM prediction probabilities."
)

if dataset_choice == "creditcard":
    st.caption(
        "💡 **Recommended Threshold: 0.80**. A 0.50 threshold yields ~95% false positive alerts (eroding trust). "
        "A 0.80 threshold maintains high recall (0.84) while improving precision (0.16)."
    )
else:
    st.caption(
        "💡 **Recommended Threshold: 0.50**. Missing a bankruptcy (false negative) causes loan default, whereas a false positive "
        "only requires manual audit. Maintaining high recall at 0.50 is optimal."
    )

s3 = get_s3_client()
prefix = f"{dataset_choice}/threshold_{threshold_choice}/"

# Check if model artifacts exist in MinIO
artifacts_exist = False
try:
    s3.head_object(Bucket='models', Key=f"{prefix}classification_report.json")
    artifacts_exist = True
except Exception:
    artifacts_exist = False

run_btn = st.button("🚀 Run Pipeline on K3s Cluster", type="primary", use_container_width=True)

if run_btn:
    endpoint = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    payload = {
        "conf": {
            "dataset": dataset_choice,
            "threshold": str(threshold_choice)
        }
    }
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
            st.toast(f"✅ Triggered Run `{run_id}`")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Failed to trigger DAG: {response.status_code}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

# --- LIVE ACTIVE EXECUTIONS SECTION ---
active_runs = get_active_dag_runs()

if active_runs:
    st.markdown("---")
    st.subheader(f"⚡ Live Cluster Executions ({len(active_runs)} Active)")
    
    current_node_pods = get_active_node_info()
    if current_node_pods:
        st.caption("📍 **Active Worker Placements**: " + " | ".join(current_node_pods))
    
    TASK_ORDER = ["init_minio_buckets", "extract_raw_data", "preprocess_and_select_features", "train_svm_dynamic"]
    
    for r in active_runs:
        run_id = r.get("dag_run_id")
        state = r.get("state", "queued").upper()
        conf = r.get("conf", {})
        ds = conf.get("dataset", "default")
        th = conf.get("threshold", "0.50")
        
        with st.expander(f"🔄 Run `{run_id}` — `{ds}` @ `{th}` | State: **{state}**", expanded=True):
            try:
                ti_resp = requests.get(
                    f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances",
                    auth=(AIRFLOW_USER, AIRFLOW_PASS),
                    timeout=4
                )
                if ti_resp.status_code == 200:
                    tis = ti_resp.json().get("task_instances", [])
                    tis.sort(key=lambda x: TASK_ORDER.index(x.get("task_id")) if x.get("task_id") in TASK_ORDER else 99)
                    
                    for ti in tis:
                        t_id = ti.get("task_id")
                        t_state = (ti.get("state") or "queued").upper()
                        t_badge = "✅" if t_state == "SUCCESS" else "❌" if t_state == "FAILED" else "⏳" if t_state == "RUNNING" else "💤"
                        st.write(f"{t_badge} `{t_id}`: **{t_state}**")
            except Exception as e:
                st.write(f"Task status error: {e}")

st.markdown("---")

# --- RESULTS & EVALUATION ---
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

if auto_refresh or active_runs:
    time.sleep(10)
    st.rerun()