# Recipe Recommendation API

A FastAPI application that provides personalized food recipe recommendations based on user preferences.

## Overview

This API uses a machine learning model to recommend food recipes based on:
- Age
- Gender
- Dietary preference (vegetarian/non-vegetarian)
- Meal type (breakfast/lunch/dinner)

## API Endpoints

- `GET /`: Health check and API information
- `GET /cuisines`: Get list of available cuisines
- `POST /recommend`: Get recipe recommendations based on user preferences
- `GET /docs/how-to-use`: Information on how to use the API
- `GET /docs`: Interactive FastAPI documentation (Swagger UI)

## Docker Deployment

The application is containerized for easy deployment:

```bash
# Build the Docker image
docker build -t recipe-recommender-api .

# Run the container
docker run -p 8000:8000 recipe-recommender-api
```

Once running, the API will be available at: http://localhost:8000

## API Usage Example

```bash
# Example API request with curl
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "female",
    "diet_preference": "vegetarian",
    "meal_type": "breakfast",
    "top_n": 5
  }'
```

## Sample Response

```json
{
  "recipes": [
    {
      "name": "Masala Dosa",
      "cooking_time": 30,
      "cuisine": "South Indian",
      "url": "https://example.com/masala-dosa"
    },
    {
      "name": "Idli Sambar",
      "cooking_time": 25,
      "cuisine": "South Indian",
      "url": "https://example.com/idli-sambar"
    },
    ...
  ],
  "count": 5
}
```

## Project Structure

- `main.py`: FastAPI application with API routes
- `model.py`: ML model logic for recipe recommendations
- `IndianFoodDatasetCSV.csv`: Dataset of food recipes
- `recipe_recommender_model.pkl`: Pre-trained recipe recommendation model

## Dependencies

- Python 3.11+
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- NumPy