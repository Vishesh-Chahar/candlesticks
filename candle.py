import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_kaggle_data():
    """Load the latest stock price dataset from Kaggle."""
    try:
        print("Loading data from Kaggle...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "nelgiriyewithana/world-stock-prices-daily-updating",
            "World-Stock-Prices-Dataset.csv",
        )
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter for a specific stock or time period if dataset is too large
        df = df.sort_values('Date', ascending=True).tail(1000)
        
        # Convert columns to numeric, replacing any non-numeric values with NaN
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove any rows with NaN values
        df = df.dropna(subset=required_columns)
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
            
        print("Data loaded successfully!")
        print("Sample of loaded data:")
        print(df.head())
        print(f"Total valid records: {len(df)}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data from Kaggle: {str(e)}")

class CandlestickDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.sequence_length = sequence_length
        
        # Ensure data is sorted by date
        if 'Date' in data.columns:
            data = data.sort_values('Date')
        
        # Calculate candlestick features
        self.data = self.calculate_candlestick_features(data)
        print(self.data)
        if len(self.data) <= sequence_length:
            raise ValueError(f"Not enough data points. Need more than {sequence_length} valid records.")
        
        # Scale the features
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        # Create sequences
        self.X, self.y = self.create_sequences()
        
        print(f"Created dataset with {len(self.X)} sequences")
    
    def calculate_candlestick_features(self, data):
        df = data.copy()
        
        # Basic candlestick features
        df['Body'] = df['Close'] - df['Open']
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 
                  'Body', 'Upper_Shadow', 'Lower_Shadow',
                  'MA5', 'MA20', 'RSI']]
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_sequences(self):
        sequences = len(self.scaled_data) - self.sequence_length
        if sequences <= 0:
            raise ValueError(f"Not enough data points to create sequences. Have {len(self.scaled_data)}, need > {self.sequence_length}")
            
        X = np.array([self.scaled_data[i:(i + self.sequence_length)] 
                     for i in range(sequences)])
        y = np.array([self.scaled_data[i + self.sequence_length, 3] 
                     for i in range(sequences)])
            
        return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.32):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # First LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Second LSTM layer - note input size is hidden_size*2 due to bidirectional first layer
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.1)  # 0.1 is the negative slope
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden states
        h0_1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0_1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        h0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0_1, c0_1))
        out = self.leaky_relu(out)  # Add activation after first LSTM
        
        out, _ = self.lstm2(out, (h0_2, c0_2))
        out = self.leaky_relu(out)  # Add activation after second LSTM
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs, progress_callback=None):
    model.train()
    training_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        
        if progress_callback:
            progress_callback(epoch, num_epochs, avg_loss)
    
    return training_losses

def save_model(model, path='stock_predictor.pth'):
    """Save the trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers
        }
    }, path)
    print(f"Model saved to {path}")

def main():
    try:
        # Load data from Kaggle
        df = load_kaggle_data()
        
        if len(df) < 100:  # Minimum data requirement
            raise ValueError("Not enough data points for meaningful training")
        
        # Create dataset
        print("Creating dataset...")
        dataset = CandlestickDataset(df)
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after preprocessing")
        
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        
        if train_size == 0 or test_size == 0:
            raise ValueError(f"Invalid split sizes: train_size={train_size}, test_size={test_size}")
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                  [train_size, test_size])
        
        # Create data loaders with smaller batch size if needed
        batch_size = min(32, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = 11  # Number of features
        hidden_size = 64
        num_layers = 2
        
        print("Initializing model...")
        model = LSTMPredictor(input_size, hidden_size, num_layers).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        
        print("Starting training...")
        training_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=1000)
        
        # Save the model
        save_model(model)
        
        return model, dataset
        
    except Exception as e:
        raise
        # return None, None
