import logging
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import h5py
from sklearn.model_selection import train_test_split

embeddings = ['Text', 'Image', 'RNA', 'Mutation']
save_path = Path("results", "recognizer", "simple")
load_path = Path("results", "recognizer", "summed_embeddings", "multi")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def build_model(input_dim):
    # Input layer
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

    # Increasing complexity for text data
    text_x = Dense(128, activation='relu', name='text_dense_1')(x)
    text_x = Dropout(0.2)(text_x)  # Adding dropout for regularization
    text_x = Dense(64, activation='relu', name='text_dense_2')(text_x)
    text_x = BatchNormalization()(text_x)
    text_x = Dropout(0.2)(text_x)  # Adding dropout for regularization
    text_x = Dense(32, activation='relu', name='text_dense_3')(text_x)
    text_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Text')(text_x)

    # Less complex paths for other outputs
    image_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Image')(x)
    rna_output = Dense(1, activation=ReLU(max_value=max_embedding), name='RNA')(x)
    mutation_output = Dense(1, activation=ReLU(max_value=max_embedding), name='Mutation')(x)

    # Separate output layers for each count
    outputs = [text_output, image_output, rna_output, mutation_output]

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='multi_output_model')
    return model


def hdf5_generator(hdf5_file_path, batch_size, indices, walk_distance):
    with h5py.File(hdf5_file_path, 'r') as f:
        X = f["X"][:]
        walk_distances = f["WalkDistances"][:] if walk_distance == -1 else None
        label_keys = [key for key in f.keys() if key not in ["X", "meta_information", "WalkDistances"]]
        labels = {key: f[key][:] for key in label_keys}
        labels = {key: labels[key] for key in embeddings}

    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = np.sort(indices[start_idx:end_idx])

            X_batch = X[batch_indices]
            y_batch = {key: labels[key][batch_indices] for key in labels}

            if walk_distance == -1:
                yield X_batch, y_batch, walk_distances[batch_indices]
            else:
                yield X_batch, y_batch


def evaluate_model_in_batches(model, generator, steps, embeddings, save_path: Path, noise: float, walk_distance: int):
    """
    Evaluate the model using a generator and save predictions, ground truth, and metrics.
    If walk_distance == -1, tracks metrics per walk distance and generates two files:
        - `metrics.csv` for aggregated results
        - `split_metrics.csv` for detailed results per walk distance
    """
    save_path.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    all_predictions = {}
    all_ground_truth = {}
    all_walk_distances_seen = set()  # Track all seen walk distances

    for _ in range(steps):
        try:
            if walk_distance == -1:
                X_batch, y_batch, walk_distance_batch = next(generator)  # ✅ Handle 3-value output
            else:
                X_batch, y_batch = next(generator)  # ✅ Handle 2-value output
                walk_distance_batch = np.full(len(X_batch), walk_distance)  # Fixed walk distance array
        except ValueError as e:
            logging.error(f"Generator error: {e}")
            continue  # Skip iteration if there's an issue

        y_pred_batch = model.predict(X_batch)

        # Track all unique walk distances seen
        all_walk_distances_seen.update(np.unique(walk_distance_batch))

        for i, embedding in enumerate(embeddings):
            y_true = y_batch[embedding]  # ✅ Use embedding names directly
            y_pred = np.rint(y_pred_batch[i])

            for wd in np.unique(walk_distance_batch):
                if wd not in all_metrics:
                    all_metrics[wd] = {emb: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for emb in embeddings}
                    all_predictions[wd] = {emb: [] for emb in embeddings}
                    all_ground_truth[wd] = {emb: [] for emb in embeddings}

                mask = walk_distance_batch == wd
                y_true_wd = y_true[mask]
                y_pred_wd = y_pred[mask]

                if len(y_true_wd) > 0:
                    all_predictions[wd][embedding].extend(y_pred_wd.flatten())
                    all_ground_truth[wd][embedding].extend(y_true_wd.flatten())

                    all_metrics[wd][embedding]['accuracy'].append(accuracy_score(y_true_wd, y_pred_wd))
                    all_metrics[wd][embedding]['precision'].append(
                        precision_score(y_true_wd, y_pred_wd, average='macro', zero_division=0))
                    all_metrics[wd][embedding]['recall'].append(
                        recall_score(y_true_wd, y_pred_wd, average='macro', zero_division=0))
                    all_metrics[wd][embedding]['f1'].append(
                        f1_score(y_true_wd, y_pred_wd, average='macro', zero_division=0))

    # ✅ Save results
    detailed_metrics = []
    aggregated_metrics = []

    if walk_distance == -1:
        for wd in sorted(all_walk_distances_seen):  # Ensure all walk distances appear
            for embedding in embeddings:
                if wd in all_metrics and embedding in all_metrics[wd]:
                    values = all_metrics[wd][embedding]
                    detailed_metrics.append({
                        "walk_distance": wd,
                        "embedding": embedding,
                        "accuracy": np.mean(values['accuracy']) if values['accuracy'] else 0,
                        "precision": np.mean(values['precision']) if values['precision'] else 0,
                        "recall": np.mean(values['recall']) if values['recall'] else 0,
                        "f1": np.mean(values['f1']) if values['f1'] else 0,
                        "noise": noise
                    })
                else:
                    detailed_metrics.append({
                        "walk_distance": wd,
                        "embedding": embedding,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "noise": noise
                    })

        split_metrics_df = pd.DataFrame(detailed_metrics)
        split_metrics_df.to_csv(Path(save_path, "split_metrics.csv"), index=False)
        logging.info(f"Detailed metrics saved to {Path(save_path, 'split_metrics.csv')}")

    # ✅ Aggregate Metrics
    for embedding in embeddings:
        all_acc = []
        all_prec = []
        all_rec = []
        all_f1 = []

        for wd in all_walk_distances_seen:
            if wd in all_metrics:
                all_acc.extend(all_metrics[wd][embedding]['accuracy'])
                all_prec.extend(all_metrics[wd][embedding]['precision'])
                all_rec.extend(all_metrics[wd][embedding]['recall'])
                all_f1.extend(all_metrics[wd][embedding]['f1'])

        aggregated_metrics.append({
            "walk_distance": -1 if walk_distance == -1 else walk_distance,
            "embedding": embedding,
            "accuracy": np.mean(all_acc) if all_acc else 0,
            "precision": np.mean(all_prec) if all_prec else 0,
            "recall": np.mean(all_rec) if all_rec else 0,
            "f1": np.mean(all_f1) if all_f1 else 0,
            "noise": noise
        })

    # ✅ Save all predictions and ground truth
    prediction_records = []
    for wd in sorted(all_walk_distances_seen):  # Iterate over all walk distances
        for embedding in embeddings:
            if wd in all_predictions and embedding in all_predictions[wd]:
                y_preds = all_predictions[wd][embedding]
                y_trues = all_ground_truth[wd][embedding]

                # Ensure y_true and y_pred lengths match
                if len(y_preds) == len(y_trues):
                    for y_true, y_pred in zip(y_trues, y_preds):
                        prediction_records.append({
                            "walk_distance": wd,
                            "embedding": embedding,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "noise": noise
                        })
                else:
                    logging.warning(f"Length mismatch for walk_distance {wd}, embedding {embedding}")

    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(prediction_records)
    predictions_df.to_csv(Path(save_path, "predictions.csv"), index=False)
    logging.info(f"Predictions saved to {Path(save_path, 'predictions.csv')}")

    metrics_df = pd.DataFrame(aggregated_metrics)
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)
    logging.info(f"Metrics saved to {Path(save_path, 'metrics.csv')}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=64, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number for the walk distance to work with.', choices=list(range(3, 31)) + [-1])
    parser.add_argument("--run_iteration", "-ri", type=int, required=False, default=1,
                        help="The iteration number for the run. Used for saving the results and validation.")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=True,
                        help="The size of the generated summed embeddings count.")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    parser.add_argument("--cancer", "-c", nargs="+", required=True,
                        help="The cancer types to work with, e.g. blca brca")
    args = parser.parse_args()

    batch_size: int = args.batch_size
    walk_distance: int = args.walk_distance
    run_iteration: int = args.run_iteration
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio
    selected_cancers: [str] = args.cancer

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers = "_".join(selected_cancers)

    logging.info("Running file simple_recognizer_nc")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Total walk distance: {walk_distance}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run iteration: {run_iteration}")
    logging.info(f"Summed embedding count: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")

    load_path: Path = Path(load_path, cancers)

    run_name: str = f"run_{run_iteration}"

    if walk_distance == -1:
        if noise_ratio == 0.0:
            train_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings.h5")
            logging.info(f"Loading data from {train_file}...")
        else:
            train_file = Path(load_path, str(amount_of_summed_embeddings), "0.0", f"combined_embeddings.h5")
            test_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings.h5")

            logging.info(f"Loading data from {train_file} and {test_file}...")

        save_path = Path(save_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings", run_name)
    else:
        if noise_ratio == 0.0:
            train_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio),
                              f"{walk_distance}_embeddings.h5")
            logging.info(f"Loading data from {train_file}...")
        else:
            # Load the test file, which is noisy
            test_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio),
                             f"{walk_distance}_embeddings.h5")
            # Load the train file, which is not noisy
            train_file = Path(load_path, str(amount_of_summed_embeddings), "0.0", f"{walk_distance}_embeddings.h5")

            logging.info(f"Loading data from {train_file} and {test_file}...")

        save_path = Path(save_path, str(amount_of_summed_embeddings), str(noise_ratio), f"{walk_distance}_embeddings",
                         run_name)

    logging.info(f"Saving results to {save_path}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Load data dimensions
    with h5py.File(train_file, "r") as f:
        input_dim = f["X"].shape[1]
        num_samples = f["X"].shape[0]
        label_keys = embeddings

    logging.info(f"Loaded HDF5 file with {num_samples} samples and input dimension {input_dim}.")

    if walk_distance != -1:
        max_embedding = walk_distance
    else:
        with h5py.File(train_file, "r") as f:
            max_embedding = f["meta_information"].attrs["max_embedding"]
            logging.info(f"Max embedding: {max_embedding}")

    logging.info("Building model....")
    model = build_model(input_dim)
    model.compile(optimizer='adam',
                  loss={'Text': 'mse', 'Image': 'mse', 'RNA': 'mse', 'Mutation': 'mse'},
                  loss_weights={'Text': 3.0, 'Image': 1., 'RNA': 1., 'Mutation': 1.},
                  metrics=['mae', 'mae', 'mae', 'mae'])
    model.summary()

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

    train_steps = max(1, len(train_indices) // batch_size)  # Ensure at least 1 step
    val_steps = max(1, len(val_indices) // batch_size)  # Ensure at least 1 step
    test_steps = max(1, len(test_indices) // batch_size)  # Ensure at least 1 step

    logging.info(f"Training on {train_steps} steps and testing on {test_steps} steps.")

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(train_gen,
                        steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=test_steps,
                        epochs=100,
                        callbacks=[early_stopping])

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "history.csv"), index=False)

    # fine tune the model
    # freeze all layer except the one containing text in the layer name
    for layer in model.layers:
        if 'Text' not in layer.name:
            layer.trainable = False

    fine_tuning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        restore_best_weights=True
    )
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    reduce_lr = ReduceLROnPlateau(monitor='val_Text_mae', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    model.compile(optimizer=optimizer,
                  loss={'Text': 'mse', 'Image': 'mse', 'RNA': 'mse', 'Mutation': 'mse'},
                  loss_weights={'Text': 4., 'Image': 0.1, 'RNA': 0.1, 'Mutation': 0.1},
                  metrics=['mae', 'mae', 'mae', 'mae'])
    model.summary()

    history = model.fit(train_gen,
                        steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=test_steps,
                        epochs=100,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Evaluate the model
    metrics: [] = evaluate_model_in_batches(model, test_gen, test_steps, embeddings=embeddings,
                                            save_path=save_path, noise=noise_ratio, walk_distance=walk_distance)

    logging.info("Done.")
