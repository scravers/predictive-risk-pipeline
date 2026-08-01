import streamlit as st
import requests
import json
import os
import boto3

# --- In-Cluster Configuration ---
# Using the internal Kubernetes DNS name instead of your NodePort IP!
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver.airflow.svc.cluster.local:8080")
DAG_ID = "financial_risk_pipeline"

# Credentials injected securely via Kubernetes Secrets
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "adminpassword")

st.set_page_config(page_title="Financial Risk Pipeline", page_icon="🏦", layout="centered")
st.title("🏦 Financial Risk ML Pipeline")
st.markdown("Trigger a distributed K3s machine learning pipeline to detect financial risks.")

# --- User Inputs ---
with st.container():
    st.subheader("1. Pipeline Configuration")
    
    dataset_choice = st.selectbox(
        "Select Dataset", 
        options=["bankruptcy", "creditcard"],
        format_func=lambda x: "Company Bankruptcy Dataset" if x == "bankruptcy" else "Credit Card Fraud Dataset"
    )
    
    threshold_choice = st.slider(
        "Select Classification Threshold", 
        min_value=0.10, max_value=0.95, value=0.50, step=0.05,
        help="Higher thresholds increase precision (fewer false positives) but lower recall."
    )

# --- Trigger Execution ---
st.subheader("2. Execution")
if st.button("🚀 Run Pipeline on K3s", type="primary"):
    
    # The Airflow REST API endpoint to trigger a DAG run
    endpoint = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    
    # The payload containing our dynamic variables
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
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                run_id = response.json().get("dag_run_id")
                st.success(f"✅ Pipeline successfully triggered! (Run ID: {run_id})")
                st.info("The pods are now spinning up on your worker nodes. Check MinIO in a few minutes for your new models and charts.")
            else:
                st.error(f"❌ Failed to trigger pipeline. Status code: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to Airflow. Please check if your NodePort IP and port are correct.")