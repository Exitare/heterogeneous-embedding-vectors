import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras import backend as ops
from tensorflow import keras
from tensorflow.keras.models import Model
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser

save_path = Path("results", "embeddings", "cancer", "new")

# Parameters
latent_dim = 768  # Dimension of the latent space
intermediate_dim = 256  # Intermediate layer size


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + (0.5 * kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == '__main__':

    if not save_path.exists():
        print(f"Creating folder {save_path}...")
        save_path.mkdir(parents=True)

    parser = ArgumentParser()
    parser.add_argument("--data", "-d", type=Path, required=True)
    parser.add_argument("--file_name", "-f", type=str, required=True)
    args = parser.parse_args()

    data_path = args.data
    file_name = args.file_name

    brca_cancer = pd.read_csv(data_path)

    input_dim = brca_cancer.shape[1]
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(brca_cancer)

    # Define the encoder
    encoder_inputs = Input(shape=(input_dim,))
    x = Dense(input_dim // 2, activation="relu")(encoder_inputs)
    x = Dense(input_dim // 3, activation="relu")(x)
    x = Dense(latent_dim, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(latent_dim, activation="relu")(latent_inputs)
    x = Dense(input_dim // 3, activation="relu")(x)
    x = Dense(input_dim // 2, activation="relu")(x)
    decoder_outputs = Dense(input_dim, activation="sigmoid")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    vae.fit(data, epochs=30, batch_size=128)

    # create embedding
    z_mean, _, _ = encoder.predict(data)
    pd.DataFrame(z_mean).to_csv(Path(save_path, file_name), index=False)
    print(f"Saved embeddings to {save_path}")
