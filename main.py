import matplotlib.pyplot as plt
from models.lstm_model import LSTM
from models.inference_engine import InferenceEngine
from training.train import load_data
import torch
from pathlib import Path 
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist, confloat
import uvicorn
import random


APP_DIR = Path(__file__).parent.absolute() 




with open(os.path.join(APP_DIR, "checkpoints/model.pkl"), "rb") as f:
        model = pickle.load(f)

lookbacks = model.lookback
app = FastAPI(
    title = "LSTM model for predicting stock prices",
    description= f"The model uses {lookbacks} lookbacks for predicting stock price",
    version="0.0.1"
)

with open(os.path.join(APP_DIR, "checkpoints", "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

# Initialize the inference engine with the pre-loaded model
engine = InferenceEngine(model)

class Item(BaseModel):
    data: conlist(confloat(gt=0), min_items=lookbacks) = Field(description="The stock prices list.", example= [round(random.uniform(50, 500), 2) for _ in range(40)])

@app.get('/', tags=['Greeting the user'])
def greeting():
    """
    Greeting the user.

    Returns:
    - dict: A welcome message.
    """
    return {"message" : "Welcome to LSTM model for predicting prices"}

@app.get('/metadata')
def get_metadata():
    """
    Retrieve the model metadata.

    Returns:
    - dict: Metadata of the model.
    """
    return metadata

@app.post("/predict")
def create_prediction(item: Item):
    """Creating predictions for data

    Args:

    - **item** (list): The data sent for making predictions. Make sure you have at least 7 observations

    Returns:
    - **predictions** (list): The predicted values 

    Example request:
        POST /predict/

        {
             "data": [357,91,222,491,172,65,446,55,461,365,188,471,352,437,51,367,462,64,309,224,225,281,284,66,109,255,460,434,250,455,299,415,481,82,398,360,177,364,428,90]
        }

    Success Response:
        Code: 200 OK
        
        Content: {
           "predictions": [292.05,290.92,295.16,300.52,286.89,290.83,291.96,281.48,286.16,292.12,272.9]
        }

    Example error request:
        POST /predict/

        {
             "data": [357,91,222,491,172,65]
        }

    Error Response:
        Code: 400 Error
        
        Content: {
           "detail": "Input data is too short. Expected at least 7 data points."
        }
    """
    raw_data = item.data
    if len(raw_data) < lookbacks:  # Check if data length is sufficient
        raise HTTPException(status_code=400, detail=f"Input data is too short. Expected at least {lookbacks} data points.")
    
    predictions = engine.run(raw_data)
    predictions_list = predictions.ravel().tolist()
    predictions = [round(i, 2) for i in predictions_list]

    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)