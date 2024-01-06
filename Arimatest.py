import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt


# Function to generate synthetic time series data
def generate_data(start_date, end_date):
    date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
    values = [i + 10 * (i % 7) + 5 * (i % 30) + 20 * (i % 365) + 10 * (i % 365) * ((i % 365) // 180) for i in
              range(len(date_rng))]
    return pd.DataFrame({'date': date_rng, 'value': values})


# Generate synthetic data from January 1, 2022, to December 31, 2023
data = generate_data('2022-01-01', '2023-12-31')

# Split data into training and validation sets
train_size = int(len(data) * 0.8)
train, validation = data[:train_size], data[train_size:]

# Define the range of values for p, d, and q
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_model = None
best_rmse = float('inf')
best_order = None  # Initialize best_order outside the loop

# Perform grid search
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)

            # Fit ARIMA model
            try:
                model = sm.tsa.ARIMA(train['value'], order=order)
                results = model.fit(disp=-1)

                # Make predictions on the validation set
                predictions = results.predict(start=len(train), end=len(train) + len(validation) - 1, dynamic=False)

                # Calculate RMSE
                rmse = sqrt(mean_squared_error(validation['value'], predictions))

                # Update best model if current one is better
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = results
                    best_order = order
            except:
                continue

# Print the best model and its order
print(f'Best ARIMA Order: {best_order}')
print(f'Best RMSE: {best_rmse}')

# You can now use 'best_model' for making predictions on future data.
