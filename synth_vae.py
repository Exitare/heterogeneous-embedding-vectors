import os, time, argparse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd
from pathlib import Path


def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return mu + K.exp(sigma / 2) * eps


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that calculates the VAE loss.
    """

    def __init__(self, feature_dim, z_mean_encoded, z_log_var_encoded, beta, **kwargs):
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.z_mean_encoded = z_mean_encoded
        self.z_log_var_encoded = z_log_var_encoded
        self.beta = beta
        self.loss_fn = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = self.feature_dim * self.loss_fn(x_input, x_decoded)
        kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var_encoded - tf.square(self.z_mean_encoded) -
                                       tf.exp(self.z_log_var_encoded), axis=-1)
        return tf.reduce_mean(reconstruction_loss + (self.beta * kl_loss))

    def call(self, inputs):
        x_input, x_decoded = inputs
        loss = self.vae_loss(x_input, x_decoded)
        self.add_loss(loss)
        return x_input


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        super(WarmUpCallback, self).__init__()
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs=None):
        if self.beta < 1:
            self.beta += self.kappa
            tf.keras.backend.set_value(self.beta, self.beta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_epochs", "-pt", type=int, default=100)
    parser.add_argument("--fine_tune_epochs", "-ft", type=int, default=100)
    parser.add_argument("--data", "-d", type=Path, required=True)

    args = parser.parse_args()

    pre_train_epochs = args.pre_train_epochs
    fine_tune_epochs = args.fine_tune_epochs
    data_path: Path = args.data

    data: pd.DataFrame = pd.read_csv(data_path)

    feature_dim = len(data.columns)
    latent_dim = 250
    batch_size = 50

    encoder_inputs = keras.Input(shape=(feature_dim,))
    z_mean_dense_linear = layers.Dense(
        latent_dim, kernel_initializer='glorot_uniform', name="encoder_1")(encoder_inputs)
    z_mean_dense_batchnorm = layers.BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = layers.Activation('relu')(z_mean_dense_batchnorm)

    z_log_var_dense_linear = layers.Dense(
        latent_dim, kernel_initializer='glorot_uniform', name="encoder_2")(encoder_inputs)
    z_log_var_dense_batchnorm = layers.BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = layers.Activation('relu')(z_log_var_dense_batchnorm)

    latent_space = layers.Lambda(
        compute_latent, output_shape=(
            latent_dim,), name="latent_space")([z_mean_encoded, z_log_var_encoded])

    decoder_to_reconstruct = layers.Dense(
        feature_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
    decoder_outputs = decoder_to_reconstruct(latent_space)

    learning_rate = 0.0005

    kappa = 1
    beta = K.variable(0)

    adam = optimizers.Adam(learning_rate=learning_rate)
    vae_layer = CustomVariationalLayer()([encoder_inputs, decoder_outputs])
    vae = Model(encoder_inputs, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    pre_train_epochs = pre_train_epochs

    fit_start = time.time()
    history = vae.fit(data,
                      epochs=pre_train_epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      callbacks=[WarmUpCallback(beta, kappa)],
                      verbose=0)

    batch_size = 10
    _ = vae.fit(data,
                epochs=fine_tune_epochs, batch_size=batch_size, shuffle=True,
                callbacks=[WarmUpCallback(beta, kappa)], verbose=0)

    encoder = Model(encoder_inputs, z_mean_encoded)
    decoder_input = keras.Input(shape=(latent_dim,))
    _x_decoded_mean = decoder_to_reconstruct(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)

    y_df = data.Labels
    decoded = pd.DataFrame(decoder.predict(encoder.predict(data)), columns=data.columns)

    latent_space = pd.DataFrame(encoder.predict(data),
                                index=data.index)

    # save latent space
    latent_space.to_csv("latent_space.csv", index=False)
