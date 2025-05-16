FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements 
COPY attached_assets/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Copy the model and dataset files
COPY attached_assets/recipe_recommender_model.pkl ./recipe_recommender_model.pkl
COPY attached_assets/IndianFoodDatasetCSV.csv ./IndianFoodDatasetCSV.csv

# Copy the application code - API only
COPY main.py ./
COPY model.py ./

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with uvicorn directly (not Gunicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]