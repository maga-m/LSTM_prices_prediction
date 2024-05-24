import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM model for time series forecasting with one LSTM layer followed by two dense layers.
    """
    def __init__(self, input_dim, hidden_dim , num_layers, output_dim, lookback = 7):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lookback = lookback
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() #hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() #cell state

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        hn = hn.view(-1, self.hidden_dim) #reshaping the data for Dense layer 
         
        out = self.fc(out[:, -1,:])
        return out

    
    