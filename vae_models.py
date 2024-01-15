import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.layers import RepeatVector
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.colors as mcolors
import seaborn as sns

LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
ARIMA_ORDER = (5, 1, 2)
ETS_SEASONAL_PERIODS = 30

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

class BaseModel:
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

class ARIMAModel(BaseModel):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def train(self, X_train, y_train):
        self.model = ARIMA(y_train, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, X_test):
        y_pred = self.model_fit.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, typ='levels')
        return y_pred

class GANModel():
    def __init__(self, epochs, batch_size, latent_dim):
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

    def train(self, X_train, y_train):
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

class ETSModel(BaseModel):
    def __init__(self, seasonal_periods):
        super().__init__()
        self.seasonal_periods = seasonal_periods

    def train(self, X_train, y_train):
        self.model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=self.seasonal_periods)
        self.model_fit = self.model.fit()

    def predict(self, X_test):
        y_pred = self.model_fit.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1)
        return y_pred

class LSTMModel(BaseModel):
    def __init__(self, epochs, batch_size):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = None
        self.scaler_y = None

    def train(self, X_train, y_train):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train.values.reshape(-1, 1))
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        model.fit(X_train_scaled, y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        self.model = model

    def predict(self, X_test):
        X_test_scaled = self.scaler_X.transform(X_test.values.reshape(-1, 1))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred.flatten()

def evaluate_model(model, X_train, y_train, X_test, y_test, params, num_runs=3):
    mse_runtimes = []

    for param_values in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), param_values))

        for _ in range(num_runs):
            model_instance = model(**param_dict)

            start_time = time.time()
            model_instance.train(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            end_time = time.time()

            mse = mean_squared_error(y_test, y_pred)
            runtime = end_time - start_time

            mse_runtimes.append((param_dict.copy(), mse, runtime))

    return mse_runtimes

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

# Define parameter grids for each model
lstm_params = {'epochs': [50, 100], 'batch_size': [32, 64]}
arima_params = {'order': [(5, 1, 2), (3, 1, 1)]}
ets_params = {'seasonal_periods': [30, 40]}
gan_params = {'epochs': [50], 'batch_size': [32], 'latent_dim': [100]}  # GAN hyperparameters

# Evaluate models with different parameters
models_to_test = {
   # 'GAN': (GANModel, gan_params),  # Include GAN in the models to test
    'LSTM': (LSTMModel, lstm_params),
    'ARIMA': (ARIMAModel, arima_params),
    'ETS': (ETSModel, ets_params),
}

mse_runtimes = []

for model_name, (model_class, model_params) in models_to_test.items():
    mse_runtime_params = evaluate_model(model_class, X_train, y_train, X_test, y_test, model_params)
    mse_runtimes.extend([(model_name, params, mse, runtime) for params, mse, runtime in mse_runtime_params])

# Create a DataFrame from the mse_runtimes list
df = pd.DataFrame(mse_runtimes, columns=['Model', 'Params', 'MSE', 'Runtime'])
# Flatten the parameters for better display
df['Params'] = df['Params'].apply(lambda x: ', '.join(f'{key}={value}' for key, value in x.items()))
print(df.sort_values(by=['Model']))
df = df.sort_values(by=['Model'])
# Create a scatter plot using Seaborn
plt.figure(figsize=(16, 8))

# Define custom color palette with varying hues for each model
custom_palette = sns.color_palette("husl", n_colors=len(df['Model'].unique()))
print(custom_palette)
# Scatter plot with transparency for individual runs
scatter = sns.scatterplot(x="Runtime", y="MSE", hue="Model", style="Params", data=df,
                          palette=custom_palette, s=100, markers=True, alpha=0.5)

# Plot means for each parameter in solid color
means_df = df.groupby(['Model', 'Params']).mean().reset_index()
means_df=means_df.sort_values(by=['Model'])
sns.scatterplot(x="Runtime", y="MSE", hue="Model", style="Params", data=means_df,
                palette=custom_palette, s=150, markers=True, legend=False)

# Set legend outside the plot
scatter.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Runtime (seconds)')
plt.ylabel('Mean Squared Error')
plt.title('Scatter Plot of MSE vs Runtime by Model and Parameters (Multiple Runs)')
plt.show()