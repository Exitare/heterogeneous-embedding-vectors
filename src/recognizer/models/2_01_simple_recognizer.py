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

embeddings = ['Text', 'Image', 'RNA', 'Mutation']
save_path = Path("results", "recognizer", "simple")
load_path = Path("results", "recognizer", "summed_embeddings", "simple")


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
    text_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_text')(text_x)

    # Less complex paths for other outputs
    image_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_image')(x)
    rna_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_rna')(x)
    mutation_output = Dense(1, activation=ReLU(max_value=max_embedding), name='output_mutation')(x)

    # Separate output layers for each count
    outputs = [text_output, image_output, rna_output, mutation_output]

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='multi_output_model')
    return model


def hdf5_generator(hdf5_file, batch_size, input_key="X", label_keys=None, start_idx=0, end_idx=None):
    """
    HDF5 generator that yields batches of data in TensorFlow-compatible format.
    """
    with h5py.File(hdf5_file, "r") as f:
        num_samples = f[input_key].shape[0]
        if end_idx is None:
            end_idx = num_samples
        while True:
            indices = np.arange(start_idx, end_idx)
            np.random.shuffle(indices)  # Shuffle the entire range of indices
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                sorted_indices = np.sort(batch_indices)  # Ensure indices are in increasing order
                X_batch = f[input_key][sorted_indices]
                y_batch = [f[key][sorted_indices] for key in label_keys]
                # Shuffle the data in memory to restore randomness
                shuffle_order = np.random.permutation(len(X_batch))
                X_batch = X_batch[shuffle_order]
                y_batch = [y[shuffle_order] for y in y_batch]
                yield X_batch, tuple(y_batch)  # Return labels as a tuple of arrays


def evaluate_model_in_batches(model, generator, steps, embeddings, save_path, noise) -> []:
    all_metrics = {embedding: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for embedding in embeddings}
    all_predictions = {embedding: [] for embedding in embeddings}
    all_ground_truth = {embedding: [] for embedding in embeddings}

    for _ in range(steps):
        X_batch, y_batch = next(generator)
        y_pred_batch = model.predict(X_batch)

        for i, embedding in enumerate(embeddings):
            y_true = y_batch[i]
            y_pred = np.rint(y_pred_batch[i])

            # Save predictions and ground truth for later CSV output
            all_predictions[embedding].extend(y_pred.flatten())
            all_ground_truth[embedding].extend(y_true.flatten())

            # Calculate metrics for the current embedding and batch
            all_metrics[embedding]['accuracy'].append(accuracy_score(y_true, y_pred))
            all_metrics[embedding]['precision'].append(
                precision_score(y_true, y_pred, average='macro', zero_division=0))
            all_metrics[embedding]['recall'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
            all_metrics[embedding]['f1'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # Average metrics across batches for each embedding
    averaged_metrics = []
    for embedding in embeddings:
        metrics = {
            "embedding": embedding,
            "accuracy": np.mean(all_metrics[embedding]['accuracy']),
            "precision": np.mean(all_metrics[embedding]['precision']),
            "recall": np.mean(all_metrics[embedding]['recall']),
            "f1": np.mean(all_metrics[embedding]['f1']),
            "noise": noise
        }
        averaged_metrics.append(metrics)

        # Save predictions and ground truth for this embedding
        pd.DataFrame({
            f'predicted_{embedding.lower()}': all_predictions[embedding],
            f'true_{embedding.lower()}': all_ground_truth[embedding]
        }).to_csv(Path(save_path, f"{embedding}_predictions.csv"), index=False)

    return averaged_metrics


if __name__ == '__main__':
    if not save_path.exists():
        save_path.mkdir(parents=True)

    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=64, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number for the walk distance to work with.')
    parser.add_argument("--run_iteration", "-ri", type=int, required=False, default=1,
                        help="The iteration number for the run. Used for saving the results and validation.")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=True,
                        help="The size of the generated summed embeddings count.")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    args = parser.parse_args()

    batch_size: int = args.batch_size
    walk_distance: int = args.walk_distance
    run_iteration: int = args.run_iteration
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio

    print(f"Total walk distance: {walk_distance}")
    print(f"Batch size: {batch_size}")
    print(f"Run iteration: {run_iteration}")
    print(f"Summed embedding count: {amount_of_summed_embeddings}")
    print(f"Noise ratio: {noise_ratio}")
    run_name = f"run_{run_iteration}"

    if walk_distance == -1:
        hdf5_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio), f"combined_embeddings.h5")
        save_path = Path(save_path, str(amount_of_summed_embeddings), str(noise_ratio), "combined_embeddings", run_name)
    else:
        hdf5_file = Path(load_path, str(amount_of_summed_embeddings), str(noise_ratio),
                         f"{walk_distance}_embeddings.h5")
        save_path = Path(save_path, str(amount_of_summed_embeddings), str(noise_ratio), f"{walk_distance}_embeddings")

    print(f"Loading data from {hdf5_file}...")
    print(f"Saving results to {save_path}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Load data dimensions
    with h5py.File(hdf5_file, "r") as f:
        input_dim = f["X"].shape[1]
        num_samples = f["X"].shape[0]
        label_keys = embeddings

    print(f"Loaded HDF5 file with {num_samples} samples and input dimension {input_dim}.")

    if walk_distance != -1:
        max_embedding = walk_distance
    else:
        with h5py.File(hdf5_file, "r") as f:
            max_embedding = f["meta_information"].attrs["max_embedding"]
            print(f"Max embedding: {max_embedding}")

    # Calculate train-test split indices
    train_end = int(0.8 * num_samples)
    test_start = train_end

    print("Building model....")
    model = build_model(input_dim)
    model.compile(optimizer='adam',
                  loss={'output_text': 'mse', 'output_image': 'mse', 'output_rna': 'mse', 'output_mutation': 'mse'},
                  loss_weights={'output_text': 3.0, 'output_image': 1., 'output_rna': 1., 'output_mutation': 1.},
                  metrics=['mae', 'mae', 'mae', 'mae'])
    model.summary()

    # Train and test generators
    train_generator = hdf5_generator(hdf5_file, batch_size, input_key="X", label_keys=label_keys, start_idx=0,
                                     end_idx=train_end)
    test_generator = hdf5_generator(hdf5_file, batch_size, input_key="X", label_keys=label_keys, start_idx=test_start,
                                    end_idx=num_samples)

    train_steps = max(1, train_end // batch_size)  # Ensure at least 1 step
    test_steps = max(1, (num_samples - train_end) // batch_size)  # Ensure at least 1 step

    print(f"Training on {train_steps} steps and testing on {test_steps} steps.")

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=test_generator,
                        validation_steps=test_steps,
                        epochs=100,
                        callbacks=[early_stopping])

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

    model.compile(optimizer=optimizer,
                  loss={'output_text': 'mse', 'output_image': 'mse', 'output_rna': 'mse', 'output_mutation': 'mse'},
                  loss_weights={'output_text': 4., 'output_image': 0.1, 'output_rna': 0.1, 'output_mutation': 0.1},
                  metrics=['mae', 'mae', 'mae', 'mae'])
    model.summary()

    history = model.fit(train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=test_generator,
                        validation_steps=test_steps,
                        epochs=100,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Evaluate the model
    metrics: [] = evaluate_model_in_batches(model, test_generator, test_steps, embeddings=embeddings, save_path=save_path, noise=noise_ratio)

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)
    print("Metrics saved.")
    print("Done.")
