import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from training.train import data_processing, split_data, load_data
import pandas.api.types as ptypes

@pytest.fixture
def sample_data():
    data = {
        'Date': pd.date_range(start='1/1/2020', periods=100),
        'Close': np.random.rand(100)
    }
    df = pd.DataFrame(data)
    return df

def test_load_data(sample_data):
    #Test the load_data function
    file_path = "tests/test_data.csv"
    sample_data.to_csv(file_path)
    df = load_data(file_path)
    assert not df.empty , "Loaded data should not be empty."


def test_data_processing(sample_data, lookback = 7, scaler = MinMaxScaler((-1,1))):
    # Test the data_processing function
    prices = sample_data['Close']
    
    sequences = data_processing(prices, lookback, scaler)
    assert sequences.shape[1] == lookback , "Each sequence should have the correct lookback length."
    assert sequences.shape[2] == 1 , "Each sequence should have one feature (price)."


def test_split_data(sample_data, lookback = 7):
    # Test the split_data function
    prices = sample_data['Close']
    dates = sample_data.index
    prices = data_processing(prices, lookback = lookback, scaler = MinMaxScaler((-1,1))) 
    x_train, y_train, x_test, y_test, train_dates, test_dates = split_data(data= prices, lookback = lookback, dates=dates)
    
    assert x_train.shape[1] == lookback - 1, "Training sequences should have lookback - 1 timesteps."
    assert y_train.shape[1] == 1, "Training targets should have one feature."
    assert len(train_dates) > len(test_dates), "Training set should be larger than the test set."
    assert isinstance(x_train, torch.Tensor), "x_train should be a torch tensor."
    assert isinstance(y_train, torch.Tensor), "y_train should be a torch tensor."
    assert isinstance(x_test, torch.Tensor), "x_test should be a torch tensor."
    assert isinstance(y_test, torch.Tensor), "y_test should be a torch tensor."




