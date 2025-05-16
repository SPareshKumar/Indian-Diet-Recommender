FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Explicitly copy individual files
COPY ./main.py ./main.py
COPY ./model.py ./model.py
COPY ./recipe_recommender_model.pkl ./recipe_recommender_model.pkl

ENV MODULE_NAME=main
