import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    balanced_accuracy_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define embeddings and paths
embeddings = ['Text', 'Image', 'RNA', 'Mutation']
save_path = Path("results", "recognizer", "multi")
load_path = Path("results", "recognizer", "summed_embeddings", "multi")


def has_valid_data(values):
    """Check if any metric in the values dictionary has data."""
    return any(len(v) > 0 for v in values.values())  # ✅ Ensure at least one metric has data


def create_indices(hdf5_file_path, walk_distance: int, test_size=0.2, random_state=42):
    """
    Create random train-test split indices with stratification based on class labels.
    """
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        num_samples = hdf5_file['X'].shape[0]
        if walk_distance == -1:
            walk_distances = hdf5_file["WalkDistances"][:]  # Load walk distances for stratification

    indices = np.arange(num_samples)

    if walk_distance != -1:
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=random_state)
        return train_indices, val_indices, test_indices

    # Stratify by walk distances
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=walk_distances
    )

    # Further split train into train and validation
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, random_state=random_state,
        stratify=walk_distances[train_indices]
    )

    return train_indices, val_indices, test_indices


def create_train_val_indices(hdf5_file_path, walk_distance: int, test_size=0.2, random_state=42):
    """
    Create random train-test split indices with stratification based on class labels.
    """
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        num_samples = hdf5_file['X'].shape[0]
        if walk_distance == -1:
            walk_distances = hdf5_file["WalkDistances"][:]  # Load walk distances for stratification

    indices = np.arange(num_samples)
    if walk_distance != -1:
        return train_test_split(indices, test_size=test_size, random_state=random_state)

        # Stratify by walk distances
    return train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=walk_distances
    )


def hdf5_generator(hdf5_file_path, batch_size, indices, walk_distance):
    with h5py.File(hdf5_file_path, 'r') as f:
        X = f["X"][:]
        walk_distances = f["WalkDistances"][:] if walk_distance == -1 else None
        label_keys = [key for key in f.keys() if key not in ["X", "meta_information", "WalkDistances"]]
        labels = {key: f[key][:] for key in label_keys}

    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = np.sort(indices[start_idx:end_idx])

            X_batch = X[batch_indices]
            y_batch = {key: labels[key][batch_indices] for key in labels}

            if walk_distance == -1:
                # logging.info(f"Generated batch with {len(X_batch)} samples and {len(y_batch)} labels.")
                yield X_batch, y_batch, walk_distances[batch_indices]
            else:
                # logging.info(f"Generated batch with {len(X_batch)} samples and {len(y_batch)} labels.")
                yield X_batch, y_batch


def evaluate_walk_distance_batches(model, generator, steps, embeddings, noise: float):
    """
    Evaluate the model in walk distance mode (walk_distance == -1).
    """

    all_predictions = []  # ✅ Store all predictions

    non_cancer_keys = {'Text', 'Image', 'RNA', 'Mutation'}
    cancer_embeddings = [emb for emb in embeddings if emb not in non_cancer_keys]

    logging.info(f"Evaluating on {steps} batches.")

    for step in range(steps):
        try:
            X_batch, y_batch, walk_distance_batch = next(generator)

            y_pred_batch = model.predict(X_batch)

            for embedding in embeddings:
                if embedding not in y_batch:
                    logging.warning(f"Missing label: {embedding}, skipping")
                    continue

                output_index = model.output_names.index(embedding)
                y_true = y_batch[embedding]
                y_pred = np.rint(y_pred_batch[output_index]).astype(int)

                if len(y_true) == 0 or len(y_pred) == 0:
                    logging.warning(f"Empty predictions or ground truth for {embedding}, skipping")
                    continue

                if y_pred.ndim > 1:
                    y_pred = y_pred.flatten()

                for wd in np.unique(walk_distance_batch):
                    mask = walk_distance_batch == wd
                    y_true_wd = y_true[mask]
                    y_pred_wd = y_pred[mask]

                    if embedding in cancer_embeddings:
                        valid_indices = np.where(y_true_wd > 0)[0]
                        if len(valid_indices) == 0:
                            continue
                        y_true_wd = y_true_wd[valid_indices]
                        y_pred_wd = y_pred_wd[valid_indices]

                    if len(y_true_wd) > 0:
                        # ✅ Store Predictions
                        for i in range(len(y_true_wd)):
                            all_predictions.append({
                                "walk_distance": wd,
                                "embedding": embedding,
                                "y_true": y_true_wd[i],
                                "y_pred": y_pred_wd[i],
                                "noise": noise
                            })

        except StopIteration:
            logging.error("Generator ran out of data earlier than expected.")
            break

    # ✅ Save Predictions
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(Path(save_path, "predictions.csv"), index=False)
        logging.info(f"Predictions saved to {Path(save_path, 'predictions.csv')}.")


def evaluate_normal_batches(model, generator, steps, embeddings, walk_distance: int, noise: float):
    """
    Evaluate the model in normal mode (walk_distance != -1).
    """

    all_predictions = []  # Store all predictions and ground truth values

    non_cancer_keys = {'Text', 'Image', 'RNA', 'Mutation'}
    cancer_embeddings = [emb for emb in embeddings if emb not in non_cancer_keys]

    logging.info(f"Evaluating normal batches using {steps} steps.")

    for step in range(steps):
        try:
            X_batch, y_batch = next(generator)

            y_pred_batch = model.predict(X_batch)

            for embedding in embeddings:
                if embedding not in y_batch:
                    logging.warning(f"Missing label: {embedding}, skipping")
                    continue

                output_index = model.output_names.index(embedding)
                y_true = y_batch[embedding]
                y_pred = np.rint(y_pred_batch[output_index]).astype(int)

                if len(y_true) == 0 or len(y_pred) == 0:
                    logging.warning(f"Empty predictions or ground truth for {embedding}, skipping")
                    continue

                # ✅ Filter out rows where cancer embeddings have y_true == 0
                if embedding in cancer_embeddings:
                    valid_indices = np.where(y_true > 0)[0]
                    if len(valid_indices) == 0:
                        logging.warning(f"Skipping {embedding} batch, no valid (non-zero) y_true values found.")
                        continue

                    y_true = y_true[valid_indices]
                    y_pred = y_pred[valid_indices]

                if y_pred.ndim > 1:
                    y_pred = y_pred.flatten()

                # ✅ Store predictions
                for i in range(len(y_true)):
                    all_predictions.append({
                        "walk_distance": walk_distance,
                        "embedding": embedding,
                        "y_true": y_true[i],
                        "y_pred": y_pred[i],
                        "noise": noise
                    })



        except StopIteration:
            logging.error("Generator ran out of data earlier than expected.")
            break

    # ✅ Save all predictions
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(Path(save_path, "predictions.csv"), index=False)
        logging.info(f"Predictions saved to {Path(save_path, 'predictions.csv')}.")


def build_model(input_dim, cancer_list: []):
    """
    Build a multi-output model for embeddings.
    """
    inputs = Input(shape=(input_dim,), name='input_layer')
    x = BatchNormalization()(inputs)
    x = Dense(512, activation='relu', name='base_dense1')(x)
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

    text_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Text')(x)
    image_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Image')(x)
    mutation_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Mutation')(x)

    rna_x = Dense(128, activation='relu', name='rna_dense1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense2')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=max_embedding), name='RNA')(rna_x)

    # Ensure cancer labels match dataset keys (no 'output_cancer_' prefix)
    cancer_outputs = [Dense(1, activation='relu', name=cancer)(x) for cancer in cancer_list]
    outputs = [text_output, image_output, mutation_output, rna_output] + cancer_outputs
    return Model(inputs=inputs, outputs=outputs, name="multi_output_model")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', "-bs", type=int, default=64, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number of the walk distance to work with.', choices=list(range(3, 1000)) + [-1])
    parser.add_argument("--run_iteration", "-ri", type=int, required=False,
                        help="The iteration number for the run. Used for saving the results and validation.", default=1)
    parser.add_argument("--cancer", "-c", nargs="+", required=True,
                        help="The cancer types to work with, e.g. BRCA LUAD STAD BLCA COAD THCA")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=True,
                        help="The amount of summed embeddings to work with.")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs to train the model")
    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    run_iteration = args.run_iteration
    selected_cancers = args.cancer
    epochs: int = args.epochs

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio
    cancers = "_".join(selected_cancers)

    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Total walk distance: {walk_distance}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run iteration: {run_iteration}")
    logging.info(f"Amount of summed embeddings: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")
    logging.info(f"Epochs: {epochs}")

    run_name = f"run_{run_iteration}"

    if walk_distance == -1:
        if noise_ratio == 0.0:
            train_file = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                              "combined_embeddings.h5")
            logging.info(f"Loading data from {train_file}")

            with h5py.File(train_file, "r") as f:
                max_embedding = f["meta_information"].attrs["max_embedding"]
                logging.info(f"Max embedding: {max_embedding}")
        else:
            train_file = Path(load_path, cancers, str(amount_of_summed_embeddings), "0.0",
                              "combined_embeddings.h5")
            test_file = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                             "combined_embeddings.h5")
            logging.info(f"Loading data from {train_file} and {test_file}")

            with h5py.File(train_file, "r") as f:
                max_embedding = f["meta_information"].attrs["max_embedding"]
                logging.info(f"Max embedding: {max_embedding}")

        save_path = Path(save_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         "combined_embeddings", run_name)
    else:
        if noise_ratio == 0.0:
            train_file = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                              f"{walk_distance}_embeddings.h5")
            logging.info(f"Loading data from {train_file}")
        else:
            train_file = Path(load_path, cancers, str(amount_of_summed_embeddings), "0.0",
                              f"{walk_distance}_embeddings.h5")
            test_file = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                             f"{walk_distance}_embeddings.h5")
            logging.info(f"Loading data from {train_file} and {test_file}")

        save_path = Path(save_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         f"{walk_distance}_embeddings", run_name)
        max_embedding = walk_distance

    logging.info(f"Saving results to {save_path}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    if noise_ratio == 0.0:
        train_indices, val_indices, test_indices = create_indices(train_file, walk_distance=walk_distance)
    else:
        train_indices, val_indices = create_train_val_indices(train_file, walk_distance=walk_distance)
        with h5py.File(test_file, "r") as f:
            test_indices = np.arange(f['X'].shape[0])

    if noise_ratio == 0.0:
        train_gen = hdf5_generator(train_file, batch_size, train_indices, walk_distance)
        val_gen = hdf5_generator(train_file, batch_size, val_indices, walk_distance)
        test_gen = hdf5_generator(train_file, batch_size, test_indices, walk_distance)
    else:
        train_gen = hdf5_generator(train_file, batch_size, train_indices, walk_distance)
        val_gen = hdf5_generator(train_file, batch_size, val_indices, walk_distance)
        test_gen = hdf5_generator(test_file, batch_size, test_indices, walk_distance)

    with h5py.File(train_file, 'r') as f:
        input_dim = f['X'].shape[1]

    model = build_model(input_dim, selected_cancers)

    # Set up a list of metrics
    loss = {'Text': 'mae', 'Image': 'mae', 'RNA': 'mae', 'Mutation': 'mae'}
    loss_weights = {'Text': 3.0, 'Image': 1., 'RNA': 1., 'Mutation': 1.}
    metrics = ['mae', 'mae', 'mae', 'mae']

    # Adding dynamic metrics for cancer outputs based on their number
    for cancer in selected_cancers:  # Assuming selected_cancers is defined
        loss[f'{cancer}'] = 'mae'
        loss_weights[f'{cancer}'] = 1.
        metrics.append('mae')
        embeddings.append(cancer)

    model.compile(optimizer='adam',
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )

    # Initial Training
    history = model.fit(train_gen, steps_per_epoch=len(train_indices) // batch_size, validation_data=val_gen,
                        validation_steps=len(val_indices) // batch_size, epochs=epochs,
                        callbacks=[early_stopping])

    # Save training history
    pd.DataFrame(history.history).to_csv(Path(save_path, "initial_training_history.csv"), index=False)

    # Fine-Tuning
    for layer in model.layers:
        if walk_distance == -1:
            key_words = selected_cancers + ["Text"]
            if layer.name not in key_words:
                layer.trainable = False
        else:
            if 'Text' not in layer.name:
                layer.trainable = False

    # adjust the text loss weight
    loss_weights["Text"] = 4.0
    loss_weights["Image"] = 0.1
    loss_weights["RNA"] = 0.1
    loss_weights["Mutation"] = 0.1
    fine_tuning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        restore_best_weights=True
    )

    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    reduce_lr = ReduceLROnPlateau(monitor='val_Text_mae', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    history = model.fit(train_gen, steps_per_epoch=len(train_indices) // batch_size, validation_data=val_gen,
                        validation_steps=len(val_indices) // batch_size, epochs=epochs,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Save fine-tuning history
    pd.DataFrame(history.history).to_csv(Path(save_path, "fine_tuning_history.csv"), index=False)

    if walk_distance != -1:
        evaluate_normal_batches(model, test_gen, len(test_indices) // batch_size, embeddings,
                                walk_distance=walk_distance, noise=noise_ratio)
    else:
        evaluate_walk_distance_batches(model, test_gen, len(test_indices) // batch_size, embeddings,
                                       noise=noise_ratio)

    logging.info("Fine-tuning and evaluation complete!")
