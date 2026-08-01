import os
import io
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import boto3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
from ucimlrepo import fetch_ucirepo 

# --- MinIO (S3) Configuration via Environment Variables ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio-svc.data-pipeline.svc.cluster.local:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")

storage_options = {
    "key": S3_ACCESS_KEY,
    "secret": S3_SECRET_KEY,
    "client_kwargs": {"endpoint_url": S3_ENDPOINT}
}

s3_client = boto3.client(
    's3', endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY
)

def extract(dataset_name):
    print(f"--- Step 1: Extracting {dataset_name} Data ---")
    
    if dataset_name == 'bankruptcy':
        dataset = fetch_ucirepo(id=572) 
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    elif dataset_name == 'creditcard':
        # Use the direct raw file URL to avoid missing fsspec/huggingface protocol dependencies
        url = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/raw/main/creditcard.csv"
        df = pd.read_csv(url)
    
    # Save to a dataset-specific path in MinIO
    s3_path = f's3://raw-data/{dataset_name}/raw.csv'
    df.to_csv(s3_path, index=False, storage_options=storage_options)
    print(f"Saved raw {dataset_name} data to {s3_path}")

def preprocess(dataset_name):
    print(f"--- Step 2: Preprocessing & Feature Selection ({dataset_name}) ---")
    
    # Read from the dataset-specific path
    df = pd.read_csv(f's3://raw-data/{dataset_name}/raw.csv', storage_options=storage_options)
    
    # Dynamically set the target column based on the dataset
    target_col = 'Bankrupt?' if dataset_name == 'bankruptcy' else 'Class'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train_unbalance, X_test, y_train_unbalance, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Downsample the majority class to match minority class
    train_data = pd.concat([X_train_unbalance, y_train_unbalance], axis=1)
    train_min = train_data[train_data[target_col] == 1]
    train_maj = train_data[train_data[target_col] == 0]
    
    train_maj_downsampled = resample(train_maj, replace=False, n_samples=len(train_min), random_state=21)
    train_balanced = pd.concat([train_maj_downsampled, train_min])
    
    X_train = train_balanced.drop(target_col, axis=1)
    y_train = train_balanced[target_col]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # L1 Feature Selection
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=21, class_weight='balanced')
    lasso.fit(X_train_scaled, y_train)
    selector = SelectFromModel(lasso, prefit=True)
    
    X_train_reduced = pd.DataFrame(selector.transform(X_train_scaled))
    X_test_reduced = pd.DataFrame(selector.transform(X_test_scaled))

    # Save to dataset-specific paths
    X_train_reduced.to_csv(f's3://processed-data/{dataset_name}/X_train.csv', index=False, storage_options=storage_options)
    X_test_reduced.to_csv(f's3://processed-data/{dataset_name}/X_test.csv', index=False, storage_options=storage_options)
    y_train.to_csv(f's3://processed-data/{dataset_name}/y_train.csv', index=False, storage_options=storage_options)
    y_test.to_csv(f's3://processed-data/{dataset_name}/y_test.csv', index=False, storage_options=storage_options)
    print(f"Preprocessing complete. Data saved to processed-data/{dataset_name}/.")

def train_and_evaluate(dataset_name, threshold):
    print(f"--- Step 3: Training & Evaluating SVM ({dataset_name} | Threshold: {threshold}) ---")
    
    # Load from dataset-specific paths
    X_train = pd.read_csv(f's3://processed-data/{dataset_name}/X_train.csv', storage_options=storage_options)
    X_test = pd.read_csv(f's3://processed-data/{dataset_name}/X_test.csv', storage_options=storage_options)
    y_train = pd.read_csv(f's3://processed-data/{dataset_name}/y_train.csv', storage_options=storage_options).squeeze()
    y_test = pd.read_csv(f's3://processed-data/{dataset_name}/y_test.csv', storage_options=storage_options).squeeze()

    param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    y_prob = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_test, y_pred))
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_name.title()} | Threshold: {threshold})')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # Include dataset name and threshold in the S3 object key
    s3_client.put_object(Bucket='models', Key=f'{dataset_name}_confusion_matrix_{threshold}.png', Body=img_buffer)
    plt.clf()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve ({dataset_name.title()})')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    s3_client.put_object(Bucket='models', Key=f'{dataset_name}_roc_curve_{threshold}.png', Body=img_buffer)
    
    # 3. Save the model object
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    s3_client.put_object(Bucket='models', Key=f'{dataset_name}_svm_model_{threshold}.pkl', Body=model_buffer)
    
    print(f"Training complete. Files uploaded to 'models' bucket with prefix '{dataset_name}_'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K3s Data Pipeline")
    parser.add_argument("--step", choices=['extract', 'preprocess', 'train'], required=True)
    parser.add_argument("--threshold", type=float, default=0.50, help="Custom threshold for classification (0.0 to 1.0)")
    parser.add_argument("--dataset", choices=['bankruptcy', 'creditcard'], default='bankruptcy', help="Which dataset to process")
    args = parser.parse_args()

    if args.step == 'extract':
        extract(args.dataset)
    elif args.step == 'preprocess':
        preprocess(args.dataset)
    elif args.step == 'train':
        train_and_evaluate(args.dataset, args.threshold)