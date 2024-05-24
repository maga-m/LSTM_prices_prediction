import pytest
import numpy as np
import torch
from models.lstm_model import LSTM
from models.inference_engine import InferenceEngine
from training.train import load_data
from sklearn.preprocessing import MinMaxScaler
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

def test_preprocess(engine):
    raw_data = torch.randn(100)  # Example raw data
    preprocessed_data = engine.preprocess(raw_data)
    assert preprocessed_data.shape[1] == engine.model.lookback, "Preprocessed data should have the correct lookback length."

def test_preprocess_value_error(engine):
    raw_data = torch.randn(5)  # Example raw data that is too short
    with pytest.raises(ValueError, match=f"Input data is too short. Expected at least {engine.lookback} data points, got {len(raw_data)}."):
        engine.preprocess(raw_data)

def test_inference(engine):
    raw_data = torch.randn(100)  # Example raw data
    predictions = engine.run(raw_data)
    assert len(predictions) == len(raw_data) - engine.model.lookback + 1, "Number of predictions should match input data length minus lookback."

