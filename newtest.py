import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

ARIMA_ORDER = (5, 1, 2)

ETS_SEASONAL_PERIODS = 30

# Function to generate synthetic data
def generate_synthetic_data(start_date, end_date, freq='D', trend_slope=0.5, amplitude=10, noise_std=5):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    time_values = np.arange(len(date_range))

    # Linear trend
    linear_trend = trend_slope * time_values

    # Seasonality (sine wave)
    seasonal_pattern = amplitude * np.sin(2 * np.pi * time_values / 365)

    # Noise
    noise = np.random.normal(0, noise_std, len(date_range))

    # Combine components to generate synthetic time series
    synthetic_data = linear_trend + seasonal_pattern + noise

    # Create a DataFrame with datetime index
    synthetic_df = pd.DataFrame({'value': synthetic_data}, index=date_range)

    return synthetic_df.sort_index()  # Sort the index to make it monotonic

def calculate_error_rate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

# Function to create error rate graphs
def plot_error_rate(models, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 6))

    for model_name, model in models.items():
        # Make predictions based on the model type
        if model_name == 'ARIMA':
            y_pred = arima_model(X_train, y_train, X_test)
        elif model_name == 'ETS':
            y_pred = ets_model(X_train, y_train, X_test)
        elif model_name == 'LSTM':
            y_pred = lstm_model(X_train, y_train, X_test)
        # Add other models as needed...

        # Calculate mean squared error
        error_rate = calculate_error_rate(y_test, y_pred)

        print(f'{model_name} - Mean Squared Error: {error_rate}')

        # Plot the error rate
        plt.plot(error_rate, label=model_name)

    plt.xlabel('Time')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error Comparison for Time Series Models')
    plt.legend()
    plt.show()

# ARIMA Model
def arima_model(X_train, y_train, X_test):
    arima_order = ARIMA_ORDER
    model = ARIMA(y_train, order=ARIMA_ORDER)

    # Fit the ARIMA model
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, typ='levels')
    return y_pred


# ETS Model
def ets_model(X_train, y_train, X_test):
    seasonal_periods = ETS_SEASONAL_PERIODS
    model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1)
    return y_pred

class GANModel():
    def __init__(self, epochs, batch_size, latent_dim):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.input_dim * K.mean(K.square(x - x_decoded_mean))
        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def build_generator(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        self.generator = model

    def build_discriminator(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))  # Adjusted this layer
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.discriminator = model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = self.generator.input
        generated_sequence = self.generator(gan_input)
        # Adjust the shape of generated_sequence
        generated_sequence = RepeatVector(self.input_dim)(generated_sequence)
        validity = self.discriminator(generated_sequence)
        self.model = Sequential([self.generator, self.discriminator])
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss=['mse', 'binary_crossentropy'])

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DatetimeIndex):
            # If X_train is a DatetimeIndex, use the index as a feature
            X_train = np.arange(len(X_train)).reshape(-1, 1)
        self.input_dim = X_train.shape[1]

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

        self.build_generator()
        self.build_discriminator()
        self.build_gan()

        for epoch in range(self.epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train_scaled.shape[0], self.batch_size)
            real_sequences = X_train_scaled[idx]
            valid = np.ones((self.batch_size, 1))
            fake_sequences = self.generator.predict(real_sequences)
            fake = np.zeros((self.batch_size, 1))
            print("Real sequences shape:", real_sequences.shape)
            print("Fake sequences shape:", fake_sequences.shape)
            d_loss_real = self.discriminator.train_on_batch(real_sequences, valid)
            fake_sequences_reshaped = fake_sequences.reshape((fake_sequences.shape[0], fake_sequences.shape[1], 1))
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences_reshaped, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = self.model.train_on_batch(real_sequences, [real_sequences, valid])


    def predict(self, X_test):
        X_test_scaled = self.scaler_X.transform(X_test.values.reshape(-1, 1))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        y_pred_scaled = self.generator.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred.flatten()

# LSTM Model with 5 layers
def lstm_model(X_train, y_train, X_test, epochs=50, batch_size=32):
    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train.values.reshape(-1, 1))
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test.values.reshape(-1, 1))

    # Reshape data for LSTM input: (samples, timesteps, features)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Define LSTM model
    model = Sequential()

    # LSTM layers
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=False))  # Last layer, return_sequences=False

    # Output layer
    model.add(Dense(1))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate as needed
    model.compile(optimizer=optimizer, loss='mse')

    # Fit the model
    model.fit(X_train_scaled, y_train_scaled, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=2)

    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform to get back to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return y_pred.flatten()


# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
start_date = '2022-01-01'
end_date = '2023-12-31'
synthetic_data = generate_synthetic_data(start_date, end_date)

# Split data into train and test sets
train_size = int(len(synthetic_data) * 0.8)
train_data, test_data = synthetic_data[:train_size], synthetic_data[train_size:]

# Define X_train, y_train, X_test, y_test
X_train, y_train = train_data.index, train_data['value']
X_test, y_test = test_data.index, test_data['value']


def evaluate_model(model_func, X_train, y_train, X_test, y_test):
    start_time = time.time()
    y_pred = model_func(X_train, y_train, X_test)
    end_time = time.time()
    mse = mean_squared_error(y_test, y_pred)
    runtime = end_time - start_time
    return mse, runtime

# Assuming you have a list of models to test
models_to_test = {'LSTM': lstm_model, 'ARIMA': arima_model, 'ETS': ets_model}

mse_runtimes = []

for model_name, model_func in models_to_test.items():
    mse, runtime = evaluate_model(model_func, X_train, y_train, X_test, y_test)
    mse_runtimes.append((model_name, mse, runtime))

# Create a scatter plot
fig, ax = plt.subplots()
for model_name, mse, runtime in mse_runtimes:
    ax.scatter(runtime, mse, label=model_name)

ax.set_xlabel('Runtime (seconds)')
ax.set_ylabel('Mean Squared Error')
ax.legend()
plt.show()