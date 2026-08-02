# To build docker buildx build --platform linux/amd64 -t scravers/predictive-risk-pipeline-frontend:v2 --load .
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir streamlit requests boto3

# Copy the Streamlit app script and optional presentation PDF
COPY app.py presentation.pdf* ./

# Streamlit runs on 8501 by default
EXPOSE 8501

# Run Streamlit headlessly (required for Docker)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]