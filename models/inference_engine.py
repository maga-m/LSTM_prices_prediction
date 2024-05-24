from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import torch



class InferenceEngine:
    """
    Engine for processing raw data, making predictions using a trained model,
    and converting predictions back to the original scale.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.lookback = model.lookback   
    
    def preprocess(self, data):
        """Normalize and reshape data for LSTM prediction.
        
        Args:
            data: raw data for the model

        Returns: 
            torch.tensor: Tensor of preprocessed data. Lookback is added and data is scaled in range of (-1, 1)
            
        """
        if len(data) < self.lookback:
            raise ValueError(f"Input data is too short. Expected at least {self.lookback} data points, got {len(data)}.")

        # Normalize data
        data = np.array(data).reshape(-1, 1)
        data = self.scaler.fit_transform(data)
        
        # Create sequences
        sequences = []
        for i in range(len(data) - self.lookback + 1):
            sequences.append(data[i:i + self.lookback])
        return torch.tensor(np.asarray(sequences), dtype=torch.float32)
    

    def postprocess(self, predictions):
        """Convert model predictions back to original scale.
        Args: 
            predictions: numpy array of predicted prices

        Returns:
            np.array: scaled predicted data
        """
        output = self.scaler.inverse_transform(predictions)
        output = np.maximum(output, 0)
        return output
        
    def predict(self, preprocessed_data):
        """Run model prediction.
        Args: 
            preprocessed_data: Toch tensor of data

        Returns:
            np.array: An array of predicted prices
        """
        with torch.no_grad():
            output = self.model(preprocessed_data)
        return output.numpy()


    def run(self, raw_data):
        """Execute the full model inference pipeline."""
        preprocessed_data = self.preprocess(raw_data)
        predictions = self.predict(preprocessed_data)
        predictions_processed = self.postprocess(predictions)
        return predictions_processed