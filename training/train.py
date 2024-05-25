import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lstm_model import LSTM
from pathlib import Path 
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as MAPE, mean_squared_error
torch.manual_seed(1)

APP_DIR = Path(__file__).parents[1]

def data_processing(data, lookback, scaler):
    """
    Generate sequences from stock prices to use as inputs for LSTM.
    Each sequence is a series of historical consecutive data points.

    Inputs:
        data: list like object that takes the data to prepare for using in LSTM
        lookback (int): number of lookbacks to use for predictions
    Returns:
        np.array: preprocessed data ready to use in LSTM model 
    """

    data = np.array(data).reshape(-1, 1)
    data = scaler.fit_transform(data)

    # create all possible sequences of length lookback
    sequence = []
    # create all possible sequences of length lookback
    for index in range(len(data) - lookback):
        sequence.append(data[index: index + lookback])
    
    sequence = np.array(sequence)
    return sequence
def split_data(data, lookback, dates):
    """
    Split the data with lookbacks into training and test sets.
    We use 90% for training and 10% for testing
    Inputs:
        stock (np.array): array with lookbacks for splitting into train and test
        lookback (int): number of lookbacks to use for predictions
        dates: list like object containing dates
    Returns:
        list: a list containing splitted data together with corresponding dates
    """


    # Define the size of the test set
    test_set_size = int(np.round(0.30 * data.shape[0]))
    
    # Ensure the test set contains the newest values
    train_set_size = data.shape[0] - test_set_size
    
    # Split the data
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    
    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]
    
    train_dates = dates[lookback:train_set_size+lookback]
    test_dates = dates[train_set_size+lookback:]
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test, train_dates, test_dates]

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(df['Date'], inplace=True, verify_integrity=False)
    return df

def main():
    """
    Loading dataset to train the model on, splitting it into train and test sets, 
    training LSTM model and saving it together with metadata
    """

    df = load_data(APP_DIR /'data/AAPL.csv')
    
    lookback = 7
    prices = df['Close']
    dates = df.index
    scaler = MinMaxScaler((-1, 1))
    prices = data_processing(prices, lookback = lookback, scaler = scaler)
    x_train, y_train, x_test, y_test, train_dates, test_dates = split_data(prices, lookback = lookback, dates = dates)
    input_dim = 1
    hidden_dim = 64
    num_layers = 1
    output_dim = 1
    lr = 0.01
    num_epochs = 600
    # Loss
    loss_fc = nn.HuberLoss()
    # Define the optimizer
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim, lookback = lookback)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hist = np.zeros(num_epochs)
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        y_train_pred = model.forward(x_train)  # set the model to training mode
        loss = loss_fc(y_train_pred, y_train)  # calculate the loss
        optimizer.zero_grad()  # zero the gradients
        loss.backward()  # backward pass
        optimizer.step()  # update the model weights   
        hist[epoch] = loss.item()     
        # Update the progress bar with the current loss
        pbar.set_postfix({"Loss": loss.item()})
        
        

    with open(APP_DIR /"checkpoints/model.pkl", "wb") as f:
        pickle.dump(model, f)

    
    with torch.no_grad():
        model.eval()  # set the model to evaluation mode
        y_test_pred = model(x_test)  # forward pass
        test_loss = loss_fc(y_test_pred, y_test)

    fig = plt.figure(figsize = (10, 6))
    plt.plot(hist, label="Training loss")
    plt.legend()
    #save the training loss image
    with open(APP_DIR /'training/loss.pkl','wb') as f:
        pickle.dump(fig, f)
    #inverse transforming results
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))#.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))#.detach().numpy())
    train_acc = MAPE(y_train, y_train_pred)
    test_acc =  MAPE(y_test, y_test_pred)
    train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_score = np.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))

    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 9))
    axes.xaxis_date()
    
    axes.plot(train_dates, y_train, color = 'red', label = 'Real AAPL Stock Price')
    axes.plot(train_dates, y_train_pred, color = 'blue', label = 'Predicted AAPL Stock Price')
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AAPLE Stock Price')
    plt.legend()
    #saving training predictions figure
    with open(APP_DIR/'training/train_pred.pkl','wb') as f:
        pickle.dump(figure, f)

    figure, axes = plt.subplots(figsize=(15, 9))
    axes.xaxis_date()
    
    axes.plot(test_dates, y_test, color = 'red', label = 'Real AAPL Stock Price')
    axes.plot(test_dates, y_test_pred, color = 'blue', label = 'Predicted AAPL Stock Price')
    plt.title('AAPL Stock Price Prediction For Testing Set')
    plt.xlabel('Time')
    plt.ylabel('AAPLE Stock Price')
    plt.legend()
    #saving test predictions figure
    with open(APP_DIR/'training/test_pred.pkl','wb') as f:
        pickle.dump(figure, f)

    


    metadata = {
    "problem": "LSTM for stock prices",
    "lookback": lookback,
    "model": "LSTM",
    "train_RMSE" : train_score,
    "test_RMSE": test_score,
    "train_MAPE" : train_acc,
    "test_MAPE": test_acc,
    "datetime": str(datetime.today().date()),
    }
    #saving metadata for API
    with open(APP_DIR/"checkpoints/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    


if __name__ == '__main__':
    main()