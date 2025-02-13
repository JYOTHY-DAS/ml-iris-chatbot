from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel #data validation library for Python.

# Load pre-trained model
with open("iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define API
app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction endpoint
@app.post("/predict/")
def predict_species(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    prediction = model.predict(data)[0]
    return {"species": prediction}
print("API created")
base_url = "http://127.0.0.1:8000"
# print(f"Default FastAPI URL is: {base_url}/")
# print(f"Here, add the endpoint /predict/: {base_url}/predict/")
# print(f"To see and test the API go to: {base_url}/docs")


# Run the API with: uvicorn iris_api:app --reload
