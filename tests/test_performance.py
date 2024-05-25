import pytest
import time
import torch
from models.lstm_model import LSTM
from models.inference_engine import InferenceEngine
from training.train import load_data
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import os
import pickle
from pathlib import Path

APP_DIR = Path(__file__).parents[1].absolute()
@pytest.fixture
def model():
    with open(os.path.join(APP_DIR, "checkpoints", "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

@pytest.fixture
def engine(model):
    return InferenceEngine(model)

def test_model_performance(engine):
    #testing model to work fast
    df = load_data(APP_DIR /'data/AAPL.csv')
    raw_data = df['Open'].values[:200]  # Sample raw data
    start_time = time.time()
    predictions = engine.run(raw_data)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 1.0, f"Inference should be fast, took {inference_time} seconds."
    assert len(predictions) == len(raw_data) - engine.model.lookback + 1, "Number of predictions should match input data length minus lookback."


def test_model_performance_acc(engine):
    #testing model accuracy
    df = load_data(APP_DIR / 'data/AAPL.csv')
    raw_data = df['Open'].values[:200]  # Sample raw data
    predictions = engine.run(raw_data)
    assert len(predictions) == len(raw_data) - engine.model.lookback + 1, "Number of predictions should match input data length minus lookback."
    
    # Performance test
    expected_values = df['Open'].values[engine.model.lookback-1:200]  # Adjust
    mape = MAPE(expected_values, predictions)
    assert mape < 10, f"Mean Squared Error should be low, got {mape}"