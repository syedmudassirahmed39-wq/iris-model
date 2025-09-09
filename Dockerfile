# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI code
COPY knn_api.py .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["python", "-m", "uvicorn", "knn_api:app", "--host", "0.0.0.0", "--port", "8000"]
