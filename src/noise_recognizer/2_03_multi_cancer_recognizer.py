import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

embeddings = ['Text', 'Image', 'RNA']
save_path = Path("results", "noise_recognizer", "multi")
clean_data_path = Path("results", "recognizer", "summed_embeddings", "multi")
noisy_data_path = Path("results", "noise_recognizer", "summed_embeddings", "multi")


def build_model(input_dim, cancer_list: []):
    inputs = Input(shape=(input_dim,), name='input_layer')
    x = Dense(512, activation='relu', name='base_dense1')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', name='base_dense2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='base_dense3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='base_dense4')(x)
    x = BatchNormalization()(x)

    # Increasing complexity for text data
    text_x = Dense(128, activation='relu', name='text_dense_1')(x)
    text_x = Dropout(0.2)(text_x)
    text_x = Dense(64, activation='relu', name='text_dense_2')(text_x)
    text_x = BatchNormalization()(text_x)
    text_x = Dropout(0.2)(text_x)
    text_x = Dense(32, activation='relu', name='text_dense_3')(text_x)
    text_output = Dense(1, activation=ReLU(max_value=walk_distance), name='output_text')(text_x)

    # Less complex paths for image output
    image_output = Dense(1, activation=ReLU(max_value=walk_distance), name='output_image')(x)

    # Path for RNA embeddings, including subtype classification
    rna_x = Dense(128, activation='relu', name='rna_dense_1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense_2')(rna_x)
    rna_x = BatchNormalization()(rna_x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(32, activation='relu', name='rna_dense_3')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=walk_distance), name='output_rna')(rna_x)

    cancer_outputs = [Dense(1, activation=ReLU(max_value=walk_distance), name=f'output_cancer_{cancer_type}')(x) for
                      cancer_type in cancer_list]

    # Combine all outputs
    outputs = [text_output, image_output, rna_output] + cancer_outputs

    # Create model
    return Model(inputs=inputs, outputs=outputs, name='multi_output_model')


if __name__ == '__main__':

    # python3 src/recognizer/multi_cancer_recognizer.py -e 5 -c blca brca
    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=32, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number of the walk distance to work with.')
    parser.add_argument("--run_iteration", "-ri", type=int, required=False,
                        help="The iteration number for the run. Used for saving the results and validation.", default=1)
    parser.add_argument("--cancer", "-c", nargs="+", required=True,
                        help="The cancer types to work with, e.g. blca brca")
    parser.add_argument("--summed_embedding_count", "-sec", type=int, required=True,
                        help="The size of the generated summed embeddings count. aka the total amount of iterations used when generating the summed embeddings.")

    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    run_iteration = args.run_iteration
    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)
    summed_embedding_count = args.summed_embedding_count

    print("Selected cancers: ", selected_cancers)
    print(f"Total walk distance: {walk_distance}")
    print(f"Batch size: {batch_size}")
    print(f"Run iteration: {run_iteration}")
    run_name = f"run_{run_iteration}"

    save_path = Path(save_path, cancers, str(summed_embedding_count))
    save_path = Path(save_path, f"{walk_distance}_embeddings")
    save_path = Path(save_path, run_name)
    print(f"Saving results to {save_path}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    load_path = Path(clean_data_path, str(summed_embedding_count), cancers, f"{walk_distance}_embeddings.csv")
    print(f"Loading data from {load_path}")

    # load data
    data = pd.read_csv(load_path)

    # Random counts for demonstration; replace with actual data
    text_counts = data["Text"].values
    image_counts = data["Image"].values
    rna_counts = data["RNA"].values

    # extract cancer data
    cancer_data = []
    for cancer in selected_cancers:
        cancer_data.append(data[cancer].values)
        embeddings.append(cancer)

    # get the number of subtypes
    X = data.drop(columns=embeddings).values

    # assert shape has 768 columns
    assert X.shape[1] == 768, f"Expected 768 features, got {X.shape[1]}"

    # Assuming these are the actual labels from your dataset
    y = [text_counts, image_counts, rna_counts] + cancer_data

    model = build_model(X.shape[1], selected_cancers)

    # Set up a list of metrics
    loss = {'output_text': 'mae', 'output_image': 'mae', 'output_rna': 'mae'}
    loss_weights = {'output_text': 3.0, 'output_image': 1., 'output_rna': 1.}
    metrics = ['mae', 'mae', 'mae']

    # Adding dynamic metrics for cancer outputs based on their number
    for i in selected_cancers:  # Assuming num_cancer_types is defined
        loss[f'output_cancer_{i}'] = 'mae'
        loss_weights[f'output_cancer_{i}'] = 1.
        metrics.append('mae')

    model.compile(optimizer='adam',
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    model.summary()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y).T, test_size=0.2, random_state=42)

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )

    # Train model
    history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                        validation_split=0.2, epochs=500, batch_size=batch_size, callbacks=[early_stopping])

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "history.csv"), index=False)

    # fine tune the model
    # freeze all layer except the one containing text in the layer name
    for layer in model.layers:
        if 'text' not in layer.name:
            layer.trainable = False

    fine_tuning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        restore_best_weights=True
    )
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    reduce_lr = ReduceLROnPlateau(monitor='val_output_text_mae', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    # adjust the text loss weight
    loss_weights["output_text"] = 4.0
    loss_weights["output_image"] = 0.1
    loss_weights["output_rna"] = 0.1

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    model.summary()
    history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                        validation_split=0.2, epochs=500, batch_size=batch_size,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Evaluate model
    results = model.evaluate(X_test, [y_test[:, i] for i in range(y_test.shape[1])])
    print("Test Loss after fine tuning:")
    print("Test MAE:", results)

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "fine_tune_history.csv"), index=False)

    # Predict counts
    y_pred = model.predict(X_test)

    # Convert predictions to rounded integers
    y_pred_rounded = [np.rint(pred) for pred in y_pred]

    # save numpy array
    np.save(Path(save_path, "y_pred.npy"), y_pred_rounded)

    # Ensure y_test is an integer array (necessary if y_test is not already in integer form)
    y_test_int = y_test.astype(int)

    # save y_test_int numpy array
    np.save(Path(save_path, "y_test.npy"), y_test_int)

    # Combined loop to create a single DataFrame for both predictions and true values for each output
    for i, embedding in enumerate(embeddings):
        embedding = embedding.lower()
        # Create a DataFrame with both predictions and true values
        df_combined = pd.DataFrame({
            f'predicted_{embedding}': y_pred_rounded[i].flatten(),
            f'true_{embedding}': y_test_int[:, i]
        })

        # convert f"predicted_{embedding}" to int
        df_combined[f'predicted_{embedding}'] = df_combined[f'predicted_{embedding}'].astype(int)

        # Save the combined DataFrame to a CSV file
        df_combined.to_csv(Path(save_path, f"combined_{embedding}_output.csv"), index=False)

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
    # calculate auc for each output
    # auc = [roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]

    # for each output, store the metrics
    for i, embedding in enumerate(embeddings):
        metrics.append({
            "walk_distance": walk_distance,
            'embedding': embedding,
            'accuracy': accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)

    print("Metrics saved.")

    # Noisy data recognition
    noisy_load_path = Path(noisy_data_path, cancers, str(summed_embedding_count))

    print("Running noise recognition...")
    noisy_metrics = []
    # running noise detection
    for noise_ratio in range(10, 110, 10):
        noise_ratio = noise_ratio / 100

        current_noise_load_path: Path = Path(noisy_load_path, f"{noise_ratio}", f"{walk_distance}_embeddings.csv")

        noisy_data = pd.read_csv(current_noise_load_path)

        noisy_truth = noisy_data[embeddings].values
        # Predict counts
        y_noise = model.predict(noisy_data.drop(columns=embeddings).values)

        # Convert predictions to rounded integers
        y_noise_rounded = [np.rint(pred) for pred in y_noise]

        noisy_accuracy = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(noisy_truth.T, y_noise_rounded)]
        # calculate f1 score for each output
        f1 = [f1_score(y_true, y_pred, average='macro') for y_true, y_pred in zip(noisy_truth.T, y_noise_rounded)]
        # calculate precision for each output
        precision = [precision_score(y_true, y_pred, average='macro') for y_true, y_pred in
                     zip(noisy_truth.T, y_noise_rounded)]
        # calculate recall for each output
        recall = [recall_score(y_true, y_pred, average='macro') for y_true, y_pred in
                  zip(noisy_truth.T, y_noise_rounded)]

        for i, embedding in enumerate(embeddings):
            noisy_metrics.append({
                "walk_distance": walk_distance,
                'embedding': embedding,
                'accuracy': noisy_accuracy[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'noise_ratio': noise_ratio
            })

    noisy_metrics_df = pd.DataFrame(noisy_metrics)
    noisy_metrics_df.to_csv(Path(save_path, "noisy_metrics.csv"), index=False)

    print("Done")
