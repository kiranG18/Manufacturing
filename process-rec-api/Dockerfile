# Use lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies (compiler tools might be needed by some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY app ./app

# Expose default port
EXPOSE 8000

# Set environment variables
ENV CLASSIFIER_MODEL_PATH="../cad-process-recommender/outputs/model_v2.pkl"
ENV COST_MODELS_DIR="../matterize-cost-model/outputs/models"
ENV LOG_LEVEL="INFO"
ENV MODEL_VERSION_TAG="classifier_v2_cost_v2"

# Start the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
