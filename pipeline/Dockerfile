# To build docker buildx build --platform linux/amd64 -t scravers/predictive-risk-pipeline:v2 --load .
# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies required for building some ML packages
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install python libraries
# Note: s3fs is required for Pandas to write to s3:// paths
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    ucimlrepo \
    boto3 \
    s3fs \
    joblib \
    datasets

# Copy your script into the container
COPY pipeline.py /app/pipeline.py

# When the container starts, it will just wait for the Airflow arguments
ENTRYPOINT ["python", "pipeline.py"]