from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import uvicorn
import os
import logging
import traceback
import sys

# Configure logging
logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

# Add handler to print logs to standard error
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter(logging_format))
logger.addHandler(console_handler)

# Log initial startup information
logger.info(f"Starting application. Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

# Import from our model file - make sure to use absolute import
model_version = "0.1.0"  # Default value in case import fails

try:
    # Import from our model file
    logger.info("Attempting to import model module...")
    from model import predict_recipes, get_model_info, get_available_cuisines, __version__ as model_version
    logger.info("Successfully imported model module")
except Exception as e:
    logger.error(f"Error importing model module: {e}")
    logger.error(traceback.format_exc())
    
    # Define fallback functions if imports fail
    def predict_recipes(age, gender, diet_preference, meal_type, top_n=8):
        logger.error("Using fallback predict_recipes function")
        return [
            {
                "name": "Fallback Recipe",
                "cooking_time": 30,
                "cuisine": "Mixed",
                "url": "https://example.com/fallback"
            }
        ]
        
    def get_model_info():
        logger.error("Using fallback get_model_info function")
        return {
            "status": "error", 
            "version": model_version,
            "meal_types": ["breakfast", "lunch", "dinner"],
            "diet_options": ["vegetarian", "non-vegetarian"]
        }
        
    def get_available_cuisines():
        logger.error("Using fallback get_available_cuisines function")
        return ["Italian", "Indian", "Mexican", "Chinese", "Thai", "American", "French", "Mediterranean"]

# Initialize FastAPI app
app = FastAPI(
    title="Recipe Recommender API",
    description="API for recommending recipes based on user preferences",
    version=model_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input/output models
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class DietPreference(str, Enum):
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non-vegetarian"

class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"

class RecipeInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age of the user")
    gender: Gender = Field(..., description="Gender of the user")
    diet_preference: DietPreference = Field(..., description="Diet preference (vegetarian/non-vegetarian)")
    meal_type: MealType = Field(..., description="Type of meal (breakfast/lunch/dinner)")
    top_n: Optional[int] = Field(8, ge=1, le=20, description="Number of recommendations to return")

class Recipe(BaseModel):
    name: str
    cooking_time: int
    cuisine: str
    url: str

class RecipeOutput(BaseModel):
    recipes: List[Recipe]
    count: int

class HealthResponse(BaseModel):
    health_check: str
    model_version: str
    model_status: str
    available_meal_types: List[str]
    available_diet_options: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

# Log startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event triggered")
    try:
        model_info = get_model_info()
        logger.info(f"Model info: {model_info}")
    except Exception as e:
        logger.error(f"Error getting model info on startup: {e}")

# Endpoints
@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint that returns the status of the API and model
    """
    logger.info("Health check endpoint called")
    try:
        model_info = get_model_info()
        
        return {
            "health_check": "OK",
            "model_version": model_version,
            "model_status": model_info["status"],
            "available_meal_types": model_info.get("meal_types", []),
            "available_diet_options": model_info.get("diet_options", [])
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "health_check": "Error",
            "model_version": model_version,
            "model_status": "error",
            "available_meal_types": ["breakfast", "lunch", "dinner"],
            "available_diet_options": ["vegetarian", "non-vegetarian"]
        }

@app.get("/cuisines", response_model=List[str])
def get_cuisines():
    """
    Return a list of available cuisines in the dataset
    """
    logger.info("Get cuisines endpoint called")
    try:
        cuisines = get_available_cuisines()
        logger.info(f"Found {len(cuisines)} cuisines")
        return cuisines
    except Exception as e:
        logger.error(f"Error retrieving cuisines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecipeOutput)
def recommend_recipes(input_data: RecipeInput):
    """
    Recommend recipes based on user preferences
    """
    logger.info(f"Recipe recommendation requested for age={input_data.age}, gender={input_data.gender}, "
                f"diet={input_data.diet_preference}, meal={input_data.meal_type}")
    try:
        # Call the predict function from model.py
        recipes = predict_recipes(
            age=input_data.age,
            gender=input_data.gender.value,
            diet_preference=input_data.diet_preference.value,
            meal_type=input_data.meal_type.value,
            top_n=input_data.top_n if input_data.top_n is not None else 8
        )
        
        logger.info(f"Returning {len(recipes)} recipe recommendations")
        return {
            "recipes": recipes,
            "count": len(recipes)
        }
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in recommend_recipes: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in recommend_recipes: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Add a documentation endpoint
@app.get("/docs/how-to-use")
def how_to_use():
    """
    Returns information on how to use the API
    """
    logger.info("How to use documentation endpoint called")
    return {
        "description": "Recipe Recommender API",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Health check and model info"
            },
            {
                "path": "/cuisines",
                "method": "GET",
                "description": "Get list of available cuisines"
            },
            {
                "path": "/recommend",
                "method": "POST",
                "description": "Get recipe recommendations",
                "request_example": {
                    "age": 30,
                    "gender": "female",
                    "diet_preference": "vegetarian",
                    "meal_type": "breakfast",
                    "top_n": 5
                }
            }
        ]
    }

# Run the API server when the script is executed directly
if __name__ == "__main__":
    # Get port from environment variable (for Heroku compatibility)
    port = int(os.environ.get("PORT", 8000))
    
    # Log the port configuration
    logger.info(f"Starting application on port {port}")
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)