# lstm_model.py
from keras.src.layers import Reshape

from vae_models import input_shape, latent_dim, batch_size, epochs, data_generator_with_trend, preprocess_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Build LSTM VAE model
def build_lstm_vae():
    # Encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = LSTM(256, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Sampling
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_h = LSTM(256, activation='relu', return_sequences=True)
    decoder_mean = LSTM(input_shape[1], activation='sigmoid', return_sequences=True)

    # Reshape z to add a timestep dimension
    z_decoded = Reshape((1, latent_dim))(z)
    h_decoded = decoder_h(z_decoded)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE Model
    vae = Model(inputs, x_decoded_mean)

    # Loss
    xent_loss = K.mean(tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae

# Train LSTM VAE model
def train_lstm_vae(train_data, validation_data):
    vae = build_lstm_vae()

    # Assuming your data has shape (batch_size, sequence_length, num_features)
    sequence_length = train_data.shape[1]
    num_features = train_data.shape[2]

    train_gen = data_generator_with_trend(batch_size, sequence_length, num_features)
    val_gen = data_generator_with_trend(batch_size, sequence_length, num_features)

    vae.fit(train_gen, steps_per_epoch=len(train_data) // batch_size, epochs=epochs,
            validation_data=val_gen, validation_steps=len(validation_data) // batch_size)
    return vae

# Evaluate LSTM VAE model
def evaluate_lstm_vae(test_data, vae):
    test_gen = data_generator_with_trend(batch_size, len(test_data))

    # Reconstruction error evaluation
    reconstruction_errors = []
    for i in range(len(test_data)//batch_size):
        batch_data, _ = next(test_gen)
        reconstructions = vae.predict(batch_data)
        mse = np.mean((batch_data - reconstructions)**2)
        reconstruction_errors.append(mse)

    return reconstruction_errors

# Plot error rate graph
def plot_error_rate_graph(reconstruction_errors):
    plt.plot(reconstruction_errors, label='Reconstruction Error')
    plt.xlabel('Batch Index')
    plt.ylabel('Error')
    plt.title('LSTM VAE Reconstruction Error')
    plt.legend()
    plt.show()

    # Assuming you have your data loaded into train_data, validation_data, and test_data


train_data = np.random.randn(1000, 100, 1)  # Replace with your actual data loading logic
validation_data = np.random.randn(200, 100, 1)  # Replace with your actual data loading logic
test_data = np.random.randn(300, 100, 1)  # Replace with your actual data loading logic

# Train LSTM VAE
lstm_vae = train_lstm_vae(train_data, validation_data)

# Evaluate LSTM VAE
reconstruction_errors = evaluate_lstm_vae(test_data, lstm_vae)

# Plot error rate graph
plot_error_rate_graph(reconstruction_errors)
