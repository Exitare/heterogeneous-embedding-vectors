import logging
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, \
    matthews_corrcoef, roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from argparse import ArgumentParser
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import h5py
from sklearn.model_selection import train_test_split
import scipy.special
from tensorflow.keras.utils import to_categorical

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
    text_output = Dense(num_classes, activation='softmax', name='Text')(text_x)

    # Less complex paths for other outputs
    image_output = Dense(num_classes, activation='softmax', name='Image')(x)
    rna_output = Dense(num_classes, activation='softmax', name='RNA')(x)
    mutation_output = Dense(num_classes, activation='softmax', name='Mutation')(x)

    # Separate output layers for each count
    outputs = [text_output, image_output, rna_output, mutation_output]

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='multi_output_model')
    return model


def hdf5_generator(hdf5_file_path, batch_size, indices, walk_distance: int):
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

            # Convert labels to one-hot encoded format
            y_batch = {key: to_categorical(labels[key][batch_indices], num_classes=num_classes) for key in labels}
            if walk_distance == -1:
                yield X_batch, y_batch, walk_distances[batch_indices]
            else:
                yield X_batch, y_batch


def evaluate_model_in_batches(model, generator, steps, embeddings, noise: float, walk_distance: int):
    """
    Evaluate the model using a generator and save predictions, ground truth, and metrics.
    If walk_distance == -1, tracks metrics per walk distance and generates two files:
        - `metrics.csv` for aggregated results
        - `split_metrics.csv` for detailed results per walk distance
    """

    all_metrics = {}
    all_predictions = {}
    all_ground_truth = {}
    all_probabilities = {}  # Store probabilities for AUC calculation
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
        y_pred_proba_batch = scipy.special.softmax(model.predict(X_batch), axis=-1)

        if y_pred_proba_batch.shape[-1] > 1:  # If model outputs logits, apply softmax
            y_pred_proba_batch = scipy.special.softmax(y_pred_proba_batch, axis=-1)

        # Track all unique walk distances seen
        all_walk_distances_seen.update(np.unique(walk_distance_batch))

        for i, embedding in enumerate(embeddings):
            y_true = y_batch[embedding]  # ✅ Use embedding names directly
            y_pred = np.rint(y_pred_batch[i])

            for wd in np.unique(walk_distance_batch):
                if wd not in all_metrics:
                    all_metrics[wd] = {
                        emb: {
                            'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                            'accuracy_zeros': [], 'precision_zeros': [], 'recall_zeros': [], 'f1_zeros': [],
                            'accuracy_nonzeros': [], 'precision_nonzeros': [], 'recall_nonzeros': [], 'f1_nonzeros': [],
                            'balanced_accuracy': [], 'mcc': [], "auc": [], "mae_zeros": [], "mae_nonzeros": [],
                            "rmse_zeros": [], "rmse_nonzeros": [], "mse_zeros": [], "mse_nonzeros": []
                        } for emb in embeddings
                    }
                    all_predictions[wd] = {emb: [] for emb in embeddings}
                    all_ground_truth[wd] = {emb: [] for emb in embeddings}
                    all_probabilities[wd] = {emb: [] for emb in embeddings}

                mask = walk_distance_batch == wd

                y_true_wd_one_hot = y_true[mask]
                y_pred_wd_one_hot = y_pred[mask]

                # Convert from one-hot encoding to categorical labels
                y_true_wd = np.argmax(y_true_wd_one_hot, axis=1)
                y_pred_wd = np.argmax(y_pred_wd_one_hot, axis=1)

                y_pred_proba_wd = y_pred_proba_batch[i][mask]  # Extract class probabilities

                if len(y_true_wd) > 0:
                    all_predictions[wd][embedding].extend(y_pred_wd.flatten())
                    all_ground_truth[wd][embedding].extend(y_true_wd.flatten())

                    # Instead of slicing, use the full probability vector.
                    y_pred_proba_full = y_pred_proba_wd  # This should always be of shape (n_samples, 4)
                    # Optionally, normalize the full vector (if not already normalized)
                    y_pred_proba_normalized = y_pred_proba_full / np.sum(y_pred_proba_full, axis=1, keepdims=True)

                    all_probabilities[wd][embedding].extend(y_pred_proba_normalized.tolist())

    # Prepare lists to accumulate records.
    prediction_records = []
    probability_records = []

    for wd in sorted(all_walk_distances_seen):  # Iterate over all walk distances
        for emb in embeddings:
            if wd in all_predictions and emb in all_predictions[wd]:
                y_preds = all_predictions[wd][emb]
                y_trues = all_ground_truth[wd][emb]

                # Ensure y_true and y_pred lengths match, then add prediction records.
                if len(y_preds) == len(y_trues):
                    for yt, yp in zip(y_trues, y_preds):
                        prediction_records.append({
                            "walk_distance": wd,
                            "embedding": emb,
                            "y_true": yt,
                            "y_pred": yp,
                            "noise": noise
                        })
                else:
                    logging.warning(f"Length mismatch for walk_distance {wd}, embedding {emb} in predictions")

                # Now add probability records if the length matches.
                if wd in all_probabilities and emb in all_probabilities[wd]:
                    prob_list = all_probabilities[wd][emb]
                    if len(prob_list) == len(y_trues):
                        for yt, yp in zip(y_trues, prob_list):
                            probability_records.append({
                                "walk_distance": wd,
                                "embedding": emb,
                                "y_true": yt,
                                "y_prob": yp,  # This is assumed to be a list of floats.
                                "noise": noise
                            })
                    else:
                        logging.warning(f"Length mismatch for walk_distance {wd}, embedding {emb} in probabilities")
                else:
                    logging.warning(f"No probabilities found for walk_distance {wd}, embedding {emb}")

    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(prediction_records)
    predictions_df.to_csv(Path(save_path, "predictions.csv"), index=False)
    logging.info(f"Predictions saved to {Path(save_path, 'predictions.csv')}")

    probabilities_df = pd.DataFrame(probability_records)
    probabilities_df.to_json(Path(save_path, "probabilities.json"), orient='records', lines=True)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=64, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=False, default=3,
                        help='The number for the walk distance to work with.', choices=list(range(3, 101)) + [-1])
    parser.add_argument("--run_iteration", "-ri", type=int, required=False, default=1,
                        help="The iteration number for the run. Used for saving the results and validation.")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=False,
                        help="The size of the generated summed embeddings count.", default=15000)
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    parser.add_argument("--cancer", "-c", nargs="+", required=False,
                        help="The cancer types to work with, e.g. blca brca",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--epochs", "-e", type=int, default=100, help="The number of epochs to train the model.")
    args = parser.parse_args()

    batch_size: int = args.batch_size
    walk_distance: int = args.walk_distance
    run_iteration: int = args.run_iteration
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio
    cancers: [str] = args.cancer
    epochs: int = args.epochs

    if len(cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        cancers = cancers[0].split(" ")

    selected_cancers = "_".join(cancers)

    logging.info("Running file simple_recognizer_nc")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Total walk distance: {walk_distance}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run iteration: {run_iteration}")
    logging.info(f"Summed embedding count: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")
    logging.info(f"Epochs: {epochs}")

    load_path: Path = Path(load_path, selected_cancers)

    run_name: str = f"run_{run_iteration}"

    if walk_distance == -1:
        if noise_ratio == 0.0:
            train_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings.h5")
            logging.info(f"Loading data from {train_file}...")
        else:
            train_file = Path(load_path, str(amount_of_summed_embeddings), "0.0", f"combined_embeddings.h5")
            test_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings.h5")

            logging.info(f"Loading data from {train_file} and {test_file}...")

        save_path = Path(save_path, selected_cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         "combined_embeddings", run_name)
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

        save_path = Path(save_path, selected_cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         f"{walk_distance}_embeddings", run_name)

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

    num_classes: int = max_embedding + 1
    logging.info(f"Max embedding: {max_embedding}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info("Building model....")
    model = build_model(input_dim)
    model.compile(optimizer='adam',
                  loss={'Text': 'categorical_crossentropy', 'Image': 'categorical_crossentropy',
                        'RNA': 'categorical_crossentropy', 'Mutation': 'categorical_crossentropy'},
                  loss_weights={'Text': 3.0, 'Image': 1., 'RNA': 1., 'Mutation': 1.},
                  metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy'])
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
                        epochs=epochs,
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
    reduce_lr = ReduceLROnPlateau(monitor='val_Text_accuracy', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    model.compile(optimizer=optimizer,
                  loss={'Text': 'categorical_crossentropy', 'Image': 'categorical_crossentropy',
                        'RNA': 'categorical_crossentropy', 'Mutation': 'categorical_crossentropy'},
                  loss_weights={'Text': 4., 'Image': 0.1, 'RNA': 0.1, 'Mutation': 0.1},
                  metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy'])
    model.summary()

    history = model.fit(train_gen,
                        steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=test_steps,
                        epochs=epochs,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Evaluate the model
    evaluate_model_in_batches(model, test_gen, test_steps, embeddings=embeddings, noise=noise_ratio, walk_distance=walk_distance)

    logging.info("Done.")
