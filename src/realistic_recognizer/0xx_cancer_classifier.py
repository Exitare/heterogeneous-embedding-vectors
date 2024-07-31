import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

load_folder = Path("results", "realistic_recognizer", "embeddings")
save_folder = Path("results", "realistic_recognizer", "predictions")

if __name__ == '__main__':
    # load embeddings
    summed_embeddings = pd.read_csv(Path(load_folder, "summed_embeddings.csv"))

    # split data into train and test using sklearn
    train, test = train_test_split(summed_embeddings, test_size=0.2)

    X_train = train.drop(columns=["submitter_id"])
    y_train = train["submitter_id"]

    X_test = test.drop(columns=["submitter_id"])
    y_test = test["submitter_id"]

    # create a classifier using tensorflow and keras
    x = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=x, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # save the model
    # model.save(Path(save_folder, "cancer_classifier.h5"))

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Loss: {loss}, Accuracy: {accuracy}")
