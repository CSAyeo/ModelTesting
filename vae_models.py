# vae_models.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Shared parameters
sequence_length = 100  # Placeholder, replace with the actual sequence length
num_features = 10  # Placeholder, replace with the actual number of features
input_shape = (sequence_length, num_features)
latent_dim = 20  # Adjust as needed
batch_size = 64
epochs = 50

# Function to preprocess the data (scaling and normalization)
def preprocess_data(data):
    # Scaling each feature to the range [0, 1]
    scaler = MinMaxScaler()
    for i in range(data.shape[-1]):
        data[:, :, i] = scaler.fit_transform(data[:, :, i])

    # Normalization (z-score normalization)
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    data = (data - mean) / std

    return data

def generate_synthetic_data_with_trend(size, noise_factor=0.5, trend_factor=0.02):
    """
    Generates synthetic time series data with noise and a random trend.

    Parameters:
    - size: The size of the generated time series.
    - noise_factor: Controls the amount of noise in the data.
    - trend_factor: Controls the strength of the random trend.

    Returns:
    - data: The generated time series data.
    """
    time = np.arange(size)
    trend = trend_factor * time
    noise = noise_factor * np.random.randn(size)
    data = trend + noise

    return data
# Generator function
def data_generator_with_trend(batch_size, sequence_length,  noise_factor=0.5, trend_factor=0.02):
    """
    Data generator function that yields batches of synthetic time series data with noise and a random trend.

    Parameters:
    - batch_size: The size of each batch.
    - size: The size of the generated time series.
    - noise_factor: Controls the amount of noise in the data.
    - trend_factor: Controls the strength of the random trend.

    Yields:
    - batch_data: A batch of generated time series data.
    """
    while True:
        batch_data = [generate_synthetic_data_with_trend(sequence_length, noise_factor, trend_factor) for _ in range(batch_size)]
        batch_data = np.array(batch_data)
        batch_data = np.expand_dims(batch_data, axis=-1)  # Add the third dimension for features
        yield batch_data, None
