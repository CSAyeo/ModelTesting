import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
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

# Evaluate models with different parameters
models_to_test = {
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

# Create a scatter plot using Seaborn
plt.figure(figsize=(12, 8))

# Define custom color palette with varying hues for each model
custom_palette = sns.color_palette("husl", n_colors=len(df['Model'].unique()))

# Scatter plot with transparency for individual runs
scatter = sns.scatterplot(x="Runtime", y="MSE", hue="Model", style="Params", data=df,
                          palette=custom_palette, s=100, markers=True, alpha=0.3)

# Plot means for each parameter in solid color
means_df = df.groupby(['Model', 'Params']).mean().reset_index()
sns.scatterplot(x="Runtime", y="MSE", hue="Model", style="Params", data=means_df,
                palette=custom_palette, s=150, markers=True, legend=False, alpha=1.0)

# Set legend outside the plot
scatter.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Runtime (seconds)')
plt.ylabel('Mean Squared Error')
plt.title('Scatter Plot of MSE vs Runtime by Model and Parameters (Multiple Runs)')
plt.show()
