from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load model once
model = joblib.load("knn_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Iris KNN API is running"}

@app.post("/predict")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(X)
    return {"predicted_class": int(prediction[0])}
