# Preserve the tested Python version.
FROM python:3.11.1-slim
WORKDIR /app
# Preserve the dependency cache until requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the deployment entry point, API and prediction helper.
COPY app/ ./app/
COPY start.py houseprice_score.py ./
# Include the fitted model only, without raw data.
COPY model_artifacts/ ./model_artifacts/
# start.py reads the platform PORT and binds to all interfaces.
CMD ["python", "start.py"]
