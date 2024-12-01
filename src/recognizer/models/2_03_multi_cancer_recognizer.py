import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import h5py

# Define embeddings and paths
embeddings = ['Text', 'Image', 'RNA', 'Mutation']
save_path = Path("results", "recognizer", "multi")
load_path = Path("results", "recognizer", "summed_embeddings", "multi")


def create_indices(hdf5_file_path, test_size=0.2, random_state=42):
    """
    Create random train-test split indices with stratification based on class labels.
    """
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        num_samples = hdf5_file['X'].shape[0]
        walk_distances = hdf5_file["WalkDistances"][:]  # Load walk distances for stratification

    indices = np.arange(num_samples)

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



def hdf5_generator(hdf5_file_path, batch_size, indices):
    """
    HDF5 data generator for summed embeddings and labels.
    """
    with h5py.File(hdf5_file_path, 'r') as f:
        X = f["X"]
        # Exclude non-datasets like "meta_information"
        label_keys = [key for key in f.keys() if key != "X" and isinstance(f[key], h5py.Dataset)]
        labels = {key: f[key] for key in label_keys}  # Map labels to their datasets

        while True:
            np.random.shuffle(indices)  # Shuffle indices for randomness
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                batch_indices = np.sort(batch_indices)  # Ensure ascending order for HDF5 compatibility
                X_batch = X[batch_indices.tolist()]  # Convert to list for HDF5 compatibility

                # Update label keys to match model output names
                y_batch = {
                    f"output_text" if key == "Text" else
                    f"output_image" if key == "Image" else
                    f"output_mutation" if key == "Mutation" else
                    f"output_rna" if key == "RNA" else
                    f"output_cancer_{key}": labels[key][batch_indices.tolist()]  # Convert to list here as well
                    for key in labels
                }

                yield X_batch, y_batch


def evaluate_model_in_batches(model, generator, steps, embeddings, save_path: Path, walk_distance: int, noise: float):
    """
    Evaluate the model using a generator and save predictions, ground truth, and metrics.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize metrics storage
    all_metrics = {embedding: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for embedding in embeddings}
    all_predictions = {embedding: [] for embedding in embeddings}
    all_ground_truth = {embedding: [] for embedding in embeddings}

    for _ in range(steps):
        X_batch, y_batch = next(generator)
        y_pred_batch = model.predict(X_batch)

        # Iterate over embeddings
        for embedding in embeddings:
            # Determine the correct output name
            if embedding in ["Text", "Image", "RNA", "Mutation"]:
                output_name = f"output_{embedding.lower()}"
            else:
                output_name = f"output_cancer_{embedding}"

            # Match predictions and ground truths for the embedding
            output_index = model.output_names.index(output_name)  # Find the correct index for this output
            y_true = y_batch[output_name]
            y_pred = np.rint(y_pred_batch[output_index]).astype(int)

            # Append predictions and ground truth
            all_predictions[embedding].extend(y_pred.flatten())
            all_ground_truth[embedding].extend(y_true.flatten())

            # Calculate and store metrics
            all_metrics[embedding]['accuracy'].append(accuracy_score(y_true, y_pred))
            all_metrics[embedding]['precision'].append(
                precision_score(y_true, y_pred, average='macro', zero_division=0))
            all_metrics[embedding]['recall'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
            all_metrics[embedding]['f1'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # Aggregate metrics for each embedding
    metrics = []
    for embedding in embeddings:
        metrics.append({
            "walk_distance": walk_distance,
            "embedding": embedding,
            "accuracy": np.mean(all_metrics[embedding]['accuracy']),
            "precision": np.mean(all_metrics[embedding]['precision']),
            "recall": np.mean(all_metrics[embedding]['recall']),
            "f1": np.mean(all_metrics[embedding]['f1']),
            "noise": noise
        })
        # Save predictions and ground truth for this embedding
        df_predictions = pd.DataFrame({
            f'predicted_{embedding.lower()}': all_predictions[embedding],
            f'true_{embedding.lower()}': all_ground_truth[embedding]
        })
        df_predictions.to_csv(Path(save_path, f"{embedding}_predictions.csv"), index=False)

    # Save overall metrics to a CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_path / "metrics.csv", index=False)
    print("Metrics saved.")
    return metrics_df


def evaluate_accuracy_per_walk_distance(metrics_df: pd.DataFrame, save_path: Path):
    """
    Calculate the accuracy per embedding and walk distance.
    """
    # Group by walk_distance and embedding to calculate mean accuracy
    grouped_metrics = metrics_df.groupby(['walk_distance', 'embedding']).agg({
        'accuracy': 'mean'
    }).reset_index()

    # Save results to CSV
    grouped_metrics.to_csv(Path(save_path, "split_metrics.csv"), index=False)
    print("Split metrics saved.")
    return grouped_metrics


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

    text_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_text')(x)
    image_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_image')(x)
    mutation_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_mutation')(x)

    rna_x = Dense(128, activation='relu', name='rna_dense1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense2')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_rna')(rna_x)

    cancer_outputs = [Dense(1, activation=ReLU(max_value=max_embedding), name=f'output_cancer_{cancer}')(x)
                      for cancer in cancer_list]
    outputs = [text_output, image_output, mutation_output, rna_output] + cancer_outputs
    return Model(inputs=inputs, outputs=outputs, name="multi_output_model")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', "-bs", type=int, default=32, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number of the walk distance to work with.')
    parser.add_argument("--run_iteration", "-ri", type=int, required=False,
                        help="The iteration number for the run. Used for saving the results and validation.", default=1)
    parser.add_argument("--cancer", "-c", nargs="+", required=True,
                        help="The cancer types to work with, e.g. blca brca")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=True,
                        help="The amount of summed embeddings to work with.")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    run_iteration = args.run_iteration
    selected_cancers = args.cancer
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio
    cancers = "_".join(selected_cancers)

    print("Selected cancers: ", selected_cancers)
    print(f"Total walk distance: {walk_distance}")
    print(f"Batch size: {batch_size}")
    print(f"Run iteration: {run_iteration}")
    run_name = f"run_{run_iteration}"

    if walk_distance == -1:
        load_path = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         "combined_embeddings.h5")
        save_path = Path(save_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings",
                         run_name)
        with h5py.File(load_path, "r") as f:
            max_embedding = f["meta_information"].attrs["max_embedding"]
            print(f"Max embedding: {max_embedding}")
    else:
        load_path = Path(load_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         f"{walk_distance}_embeddings.h5")
        save_path = Path(save_path, cancers, str(amount_of_summed_embeddings), str(noise_ratio),
                         f"{walk_distance}_embeddings", run_name)
        max_embedding = walk_distance

    print(f"Loading data from {load_path}")
    print(f"Saving results to {save_path}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    train_indices, val_indices, test_indices = create_indices(load_path)

    train_gen = hdf5_generator(load_path, batch_size, train_indices)
    val_gen = hdf5_generator(load_path, batch_size, val_indices)
    test_gen = hdf5_generator(load_path, batch_size, test_indices)

    with h5py.File(load_path, 'r') as f:
        input_dim = f['X'].shape[1]

    model = build_model(input_dim, selected_cancers)
    # Set up a list of metrics
    loss = {'output_text': 'mae', 'output_image': 'mae', 'output_rna': 'mae', 'output_mutation': 'mae'}
    loss_weights = {'output_text': 3.0, 'output_image': 1., 'output_rna': 1., 'output_mutation': 1.}
    metrics = ['mae', 'mae', 'mae', 'mae']

    # Adding dynamic metrics for cancer outputs based on their number
    for cancer in selected_cancers:  # Assuming num_cancer_types is defined
        loss[f'output_cancer_{cancer}'] = 'mae'
        loss_weights[f'output_cancer_{cancer}'] = 1.
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
                        validation_steps=len(val_indices) // batch_size, epochs=100,
                        callbacks=[early_stopping])

    # Save training history
    pd.DataFrame(history.history).to_csv(Path(save_path, "initial_training_history.csv"), index=False)

    # Fine-Tuning
    for layer in model.layers:
        if 'text' not in layer.name:
            layer.trainable = False

    # adjust the text loss weight
    loss_weights["output_text"] = 4.0
    loss_weights["output_image"] = 0.1
    loss_weights["output_rna"] = 0.1
    loss_weights["output_mutation"] = 0.1
    fine_tuning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        restore_best_weights=True
    )

    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    reduce_lr = ReduceLROnPlateau(monitor='val_output_text_mae', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    history = model.fit(train_gen, steps_per_epoch=len(train_indices) // batch_size, validation_data=val_gen,
                        validation_steps=len(val_indices) // batch_size, epochs=100,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Save fine-tuning history
    pd.DataFrame(history.history).to_csv(save_path / "fine_tuning_history.csv", index=False)

    # Final Evaluation
    metrics_df = evaluate_model_in_batches(model, test_gen, len(test_indices) // batch_size, embeddings, save_path,
                                           walk_distance=walk_distance, noise=noise_ratio)

    # Calculate and save accuracy per embedding and walk distance
    accuracy_metrics_df = evaluate_accuracy_per_walk_distance(metrics_df=metrics_df, save_path=save_path)
    print("Fine-tuning and evaluation complete!")
