import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

embeddings = ['Text', 'Image', 'RNA']
save_path = Path("results","recognizer")
load_path = Path("results")


def build_model(input_dim, num_outputs=3):
    # Input layer
    inputs = Input(shape=(input_dim,), name='input_layer')

    # Common base layers
    x = Dense(128, activation='relu', name='base_dense1')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='base_dense2')(x)
    x = BatchNormalization()(x)

    # Separate output layers for each count
    outputs = []
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='relu', name=f'output_{i}')(x))

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='multi_output_model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mae', 'mae'])

    return model


if __name__ == '__main__':
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load data
    data = pd.read_csv(Path(load_path, "summed_embeddings.csv"))

    # Random counts for demonstration; replace with actual data
    text_counts = data["Text"].values
    image_counts = data["Image"].values
    rna_counts = data["RNA"].values

    X = data.drop(columns=["Text", "Image", "RNA"]).values

    # Assuming these are the actual labels from your dataset
    y = [text_counts, image_counts, rna_counts]

    model = build_model(X.shape[1])
    model.summary()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y).T, test_size=0.2, random_state=42)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    # Train model
    history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                        validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stopping])

    # Evaluate model
    results = model.evaluate(X_test, [y_test[:, i] for i in range(y_test.shape[1])])
    print("Test Loss, Test MAE:", results)

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "history.csv"), index=False)

    # Predict counts
    y_pred = model.predict(X_test)

    # Convert predictions to rounded integers
    y_pred_rounded = [np.round(pred) for pred in y_pred]
    # save numpy array
    np.save(Path(save_path, "y_pred_rounded.npy"), y_pred_rounded)

    # Ensure y_test is an integer array (necessary if y_test is not already in integer form)
    y_test_int = y_test.astype(int)
    # save y_test_int numpy array
    np.save(Path(save_path, "y_test_int.npy"), y_test_int)

    # save predictions for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_pred_rounded[i]).to_csv(Path(save_path, f"predictions_{embedding}_output.csv"),
                                               index=False)

    # save y_test for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_test_int[:, i]).to_csv(Path(save_path, f"true_{embedding}_output.csv"),
                                              index=False)

    metrics = []
    # Calculate accuracy for each output
    accuracy = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]
    # calculate f1 score for each output
    f1 = [f1_score(y_true, y_pred, average='macro') for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]
    # calculate precision for each output
    precision = [precision_score(y_true, y_pred, average='macro') for y_true, y_pred in
                 zip(y_test_int.T, y_pred_rounded)]
    # calculate recall for each output
    recall = [recall_score(y_true, y_pred, average='macro') for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]

    # for each output, store the metrics
    for i, embedding in enumerate(embeddings):
        metrics.append({
            'embedding': embedding,
            'accuracy': accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)

    # Convert predictions to binary
    y_pred_binary = [np.where(pred > 0.5, 1, 0) for pred in y_pred]  # Adjust 0.5 based on your thresholding needs
    # save numpy array
    np.save(Path(save_path, "y_pred_binary.npy"), y_pred_binary)

    # Convert true values to binary
    y_test_binary = [np.where(true > 0, 1, 0) for true in y_test.T]
    # save numpy array
    np.save(Path(save_path, "y_test_binary.npy"), y_test_binary)

    # save binary predictions for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_pred_binary[i]).to_csv(Path(save_path, f"binary_predictions_{embedding}_output.csv"),
                                              index=False)

    # save binary true values for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_test_binary[i]).to_csv(Path(save_path, f"binary_true_{embedding}_output.csv"),
                                              index=False)

    binary_metrics = []
    # calculate accuracy
    accuracy = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]
    # calculate precision
    precision = [precision_score(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]
    # calculate recall
    recall = [recall_score(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]
    # calculate f1 score
    f1 = [f1_score(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]

    # for each output, store the metrics
    for i, embedding in enumerate(embeddings):
        binary_metrics.append({
            'embedding': embedding,
            'accuracy': accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        })

    binary_metrics_df = pd.DataFrame(binary_metrics)
    binary_metrics_df.to_csv(Path(save_path, "binary_metrics.csv"), index=False)
