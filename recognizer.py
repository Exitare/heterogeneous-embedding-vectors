import tensorflow as tf
import pandas as pd
from pathlib import Path


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='sigmoid')  # Three output units for three labels
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # load data


    model = build_model(500)
    model.summary()