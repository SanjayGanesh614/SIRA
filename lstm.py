import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Dataset Class for time series windows
class OceanDataset(Dataset):
    def __init__(self, data, window_size):
        """
        data: numpy array of shape (samples, features)
        window_size: length of time series windows
        """
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]
        return torch.tensor(window, dtype=torch.float32)

# 2. LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, feature_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        _, (h, c) = self.encoder(x)  # h shape: (num_layers, batch, hidden_dim)
        decoder_input = h.repeat(seq_len, 1, 1).permute(1, 0, 2).contiguous()
        out, _ = self.decoder(decoder_input)
        return out

# 3. Training Function
def train_model(model, dataloader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# 4. Calculate Reconstruction Error & Detect Anomalies
def detect_anomalies(model, dataloader, threshold=None):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            mse = ((batch - output) ** 2).mean(dim=[1, 2])  # error per sequence
            errors.extend(mse.cpu().numpy())
    errors = np.array(errors)
    if threshold is None:
        threshold = np.percentile(errors, 95)  # default threshold as 95th percentile
    anomaly_indices = np.where(errors > threshold)[0]
    return anomaly_indices, errors, threshold

# 1a to 1c: Extended anomaly localization - pinpoint time step and variable causing max reconstruction error
def localize_anomalies(model, dataloader, anomaly_indices):
    """
    For each anomaly window index, compute reconstruction error matrix and localize max error.

    Returns:
        List of dicts: [{'window_idx': int, 'max_error': float, 'time_idx': int, 'var_idx': int}, ...]
    """
    model.eval()
    localizations = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i not in anomaly_indices:
                continue  # process only detected anomaly windows

            output = model(batch)  # same shape as batch: (1, seq_len, feat_dim)
            error_matrix = (batch - output) ** 2  # element-wise squared errors

            # Remove batch dimension
            error_matrix = error_matrix[0]  # shape: (seq_len, feat_dim)

            # Find max error and its location
            max_error_val = error_matrix.max().item()
            max_error_idx = error_matrix.argmax().item()
            seq_len, feat_dim = error_matrix.shape
            time_idx = max_error_idx // feat_dim
            var_idx = max_error_idx % feat_dim

            localization_info = {
                'window_idx': i,
                'max_error': max_error_val,
                'time_idx': time_idx,
                'var_idx': var_idx
            }
            localizations.append(localization_info)

    return localizations


if __name__ == "__main__":
    # === Data Loading and Preprocessing ===
    df = pd.read_csv("biogeodata.csv", skiprows=[1])  # Skip the units row

    # Select variables relevant for anomaly detection
    selected_vars = [
        'Temperature',
        'CTD_Salinity',
        'Oxygen_1',
        'CO2',
        'Alkalinity',
        'TOC',
        'POC',
        'NO3_plus_NO2',
        'Silicate',
        'Bact_Enum'
    ]

    # Extract, convert to numeric, and fill NaN values by forward/backward fill
    data = df[selected_vars].apply(pd.to_numeric, errors='coerce').ffill().bfill()

    # Convert to NumPy array
    data_array = data.values

    # Normalize features with StandardScaler
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data_array)

    print("Data normalized with shape:", data_norm.shape)

    # Create sliding windows dataset and dataloader
    window_size = 10
    dataset = OceanDataset(data_norm, window_size)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the LSTM Autoencoder
    model = LSTMAutoencoder(feature_dim=len(selected_vars), hidden_dim=64, num_layers=1)

    # Train the model
    train_model(model, dataloader, epochs=30, lr=0.001)

    # Use the whole dataset for anomaly detection
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    anomalies, errors, threshold = detect_anomalies(model, test_loader)

    print(f"Anomaly threshold: {threshold}")
    print(f"Anomalies detected at indices: {anomalies}")

    # Localize anomalies within detected anomaly windows
    localizations = localize_anomalies(model, test_loader, anomalies)
    for loc in localizations[:10]:  # print first 10 localization results
        print(f"Window {loc['window_idx']} - Max error {loc['max_error']:.4f} at time step {loc['time_idx']}, variable index {loc['var_idx']}")

    # Plot reconstruction errors and threshold
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(threshold, color='r', linestyle='--', label="Anomaly Threshold")
    plt.xlabel("Sequence Index")
    plt.ylabel("Mean Squared Reconstruction Error")
    plt.title("Anomaly Detection from LSTM Autoencoder Reconstruction Error")
    plt.legend()
    plt.show()
