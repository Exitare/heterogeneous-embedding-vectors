import os, time, argparse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf
import pandas as pd
from pathlib import Path

save_folder = Path("results", "embeddings")


def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return mu + K.exp(sigma / 2) * eps


class CustomVariationalLayer(Layer):
    """
    Define a custom layer for Variational Autoencoder with loss calculation
    """

    def __init__(self, **kwargs):
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded, z_mean_encoded, z_log_var_encoded, beta):
        reconstruction_loss = feature_dim * tf.keras.losses.binary_crossentropy(x_input, x_decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
        total_loss = K.mean(reconstruction_loss + (beta * kl_loss))
        return total_loss

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        z_mean_encoded = inputs[2]
        z_log_var_encoded = inputs[3]
        beta = inputs[4]
        loss = self.vae_loss(x, x_decoded, z_mean_encoded, z_log_var_encoded, beta)
        self.add_loss(loss)
        return x_decoded


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        super(WarmUpCallback, self).__init__()
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs=None):
        new_beta = K.get_value(self.beta) + self.kappa
        K.set_value(self.beta, new_beta if new_beta < 1 else 1)


if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_epochs", "-pt", type=int, default=100)
    parser.add_argument("--fine_tune_epochs", "-ft", type=int, default=100)
    parser.add_argument("--data", "-d", type=Path, required=True)

    args = parser.parse_args()

    pre_train_epochs = args.pre_train_epochs
    fine_tune_epochs = args.fine_tune_epochs
    data_path: Path = args.data

    data: pd.DataFrame = pd.read_csv(data_path, sep="\t", index_col=0)

    # check that all columns to be float
    for column in data.columns:
        if data[column].dtype != float:
            print(f"{column} is not float")

    feature_dim = len(data.columns)
    latent_dim = 384
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
    vae_layer = CustomVariationalLayer()([encoder_inputs, decoder_outputs, z_mean_encoded, z_log_var_encoded, beta])
    vae = Model(encoder_inputs, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    pre_train_epochs = pre_train_epochs

    fit_start = time.time()
    history = vae.fit(data,
                      epochs=pre_train_epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      callbacks=[WarmUpCallback(beta, kappa)],
                      verbose=1)

    batch_size = 10
    _ = vae.fit(data,
                epochs=fine_tune_epochs, batch_size=batch_size, shuffle=True,
                callbacks=[WarmUpCallback(beta, kappa)], verbose=1)

    encoder = Model(encoder_inputs, z_mean_encoded)
    decoder_input = keras.Input(shape=(latent_dim,))
    _x_decoded_mean = decoder_to_reconstruct(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)

    decoded = pd.DataFrame(decoder.predict(encoder.predict(data)), columns=data.columns)

    latent_space = pd.DataFrame(encoder.predict(data), index=data.index)

    # save latent space
    latent_space.to_csv(Path(save_folder, "rna_embeddings.csv"), index=False)
