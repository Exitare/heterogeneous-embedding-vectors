import h5py
import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
from collections import Counter
import math

load_folder = Path("results", "classifier", "summed_embeddings")
save_folder = Path("results", "classifier", "classification")


def h5_generator_specific_indices(h5_file_path, indices, batch_size, label_encoder):
    """
    Generator that yields batches of data based on specific indices.
    """
    num_samples = len(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        with h5py.File(h5_file_path, 'r') as h5_file:
            X_batch = h5_file["X"][batch_indices]
            y_batch = h5_file["y"][batch_indices]

        # Decode and encode labels
        y_batch = [label.decode("utf-8") for label in y_batch]
        y_batch = label_encoder.transform(y_batch)
        yield X_batch, y_batch


def create_tf_dataset_specific_indices(h5_file_path, indices, batch_size, label_encoder):
    """
    Creates a TensorFlow dataset from specific indices.
    """
    generator = lambda: h5_generator_specific_indices(h5_file_path, indices, batch_size, label_encoder)
    # Retrieve feature dimension from the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        feature_dimension = h5_file.attrs["feature_shape"]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, feature_dimension), dtype=tf.float32),  # Features
            tf.TensorSpec(shape=(None,), dtype=tf.int32)  # Labels
        )
    )
    return dataset


def train_and_evaluate_model(train_ds, val_ds, test_ds, num_classes: int, save_folder: Path, iteration: int,
                             walk_distance: int,
                             amount_of_walks: int, label_encoder):
    input_layer = tf.keras.layers.Input(shape=(train_ds.element_spec[0].shape[1],))

    x = BatchNormalization()(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(train_ds,
                        epochs=50,
                        steps_per_epoch=train_batches,
                        validation_data=val_ds,
                        validation_steps=val_batches,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        ])

    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Predict and collect results
    y_test = []
    y_pred = []
    y_pred_proba = []  # Collect probabilities instead of labels
    for X_batch, y_batch in test_ds:
        y_test.extend(y_batch.numpy())
        y_pred.extend(model.predict(X_batch).argmax(axis=1))
        y_pred_proba.extend(model.predict(X_batch))

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    predictions = pd.DataFrame({
        "y_test": y_test,
        "y_test_decoded": label_encoder.inverse_transform(y_test),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred)
    })

    predictions.to_csv(Path(save_folder, f"predictions.csv"), index=False)

    # Metrics by cancer type
    results = []
    for cancer in np.unique(y_test):
        y_test_cancer = y_test[y_test == cancer]
        y_pred_cancer = y_pred[y_test == cancer]
        cancer_name = label_encoder.inverse_transform([cancer])[0]

        accuracy_cancer = (y_test_cancer == y_pred_cancer).mean()
        f1_cancer = f1_score(y_test_cancer, y_pred_cancer, average='weighted')
        precision_cancer = precision_score(y_test_cancer, y_pred_cancer, average='weighted')
        recall_cancer = recall_score(y_test_cancer, y_pred_cancer, average='weighted')

        print(
            f"{cancer_name}: Accuracy: {accuracy_cancer:.4f}, F1: {f1_cancer:.4f}, Precision: {precision_cancer:.4f}, Recall: {recall_cancer:.4f}"
        )

        results.append({
            "cancer": cancer_name,
            "accuracy": accuracy_cancer,
            "f1": f1_cancer,
            "precision": precision_cancer,
            "recall": recall_cancer,
            "iteration": iteration,
            "walk_distance": walk_distance,
            "amount_of_walks": amount_of_walks
        })

    # Overall metrics
    f1_total = f1_score(y_test, y_pred, average='weighted')
    precision_total = precision_score(y_test, y_pred, average='weighted')
    recall_total = recall_score(y_test, y_pred, average='weighted')
    accuracy_total = (y_test == y_pred).mean()

    print(y_test.shape)
    print(y_pred.shape)
    # Assuming the number of classes is known
    num_classes = y_pred_proba.shape[1]  # Infer from probabilities
    y_test_one_hot = label_binarize(y_test, classes=np.arange(num_classes))
    # Compute AUC-ROC score
    auc_score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovo', average='macro')

    print(
        f"Overall: Accuracy: {accuracy_total:.4f}, F1: {f1_total:.4f}, Precision: {precision_total:.4f}, Recall: {recall_total:.4f}, AUC: {auc_score:.4f}")

    results.append({
        "cancer": "All",
        "accuracy": accuracy_total,
        "f1": f1_total,
        "precision": precision_total,
        "recall": recall_total,
        "auc": auc_score,
        "iteration": iteration,
        "walk_distance": walk_distance,
        "amount_of_walks": amount_of_walks
    })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(save_folder, f"results.csv"), index=False)
    print("Results saved.")

    # Save model and training history
    model.save(Path(save_folder, f"model.h5"))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(save_folder, f"history.csv"), index=False)
    print("Model and history saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--iteration", "-i", type=int, required=True, help="The iteration number.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.", choices=[3, 4, 5],
                        default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5], default=3)
    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    walk_amount = args.amount_of_walks
    iteration = args.iteration

    cancers = "_".join(args.cancer)
    print(f"Selected cancers: {cancers}")
    print(f"Walk distance: {walk_distance}, Amount of walks: {walk_amount}")

    load_folder = Path(load_folder, cancers, f"{walk_distance}_{walk_amount}")
    h5_file_path = Path(load_folder, "summed_embeddings.h5")

    cancer_save_folder = Path(save_folder, cancers)
    cancer_save_folder = Path(cancer_save_folder, f"{walk_distance}_{walk_amount}")
    iteration_save_folder = Path(cancer_save_folder, str(iteration))

    if not cancer_save_folder.exists():
        cancer_save_folder.mkdir(parents=True)

    if not iteration_save_folder.exists():
        iteration_save_folder.mkdir(parents=True)

    train_ratio = 0.7
    val_ratio = 0.05
    test_ratio = 0.25

    class_counts = Counter()

    with h5py.File(h5_file_path, "r") as h5_file:
        y = h5_file["y"]
        chunk_size = 10000  # Process data in chunks
        total_samples = y.shape[0]

        for i in range(0, total_samples, chunk_size):
            chunk = y[i:i + chunk_size]
            # convert chunks to utf-8
            chunk = [label.decode("utf-8") for label in chunk]
            class_counts.update(chunk)

    split_sizes = {cls: {
        "train": int(count * train_ratio),
        "val": int(count * val_ratio),
        "test": count - int(count * train_ratio) - int(count * val_ratio)
    } for cls, count in class_counts.items()}

    # To store indices
    split_indices = {"train": [], "val": [], "test": []}
    allocated = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    chunk_size = 10000
    with h5py.File(h5_file_path, "r") as h5_file:
        y = h5_file["y"]
        feature_dimension = h5_file.attrs["feature_shape"]
        unique_classes = h5_file.attrs["classes"]
        for i in range(0, y.shape[0], chunk_size):
            chunk = y[i:i + chunk_size]  # Load a chunk
            chunk_indices = np.arange(i, i + len(chunk))  # Indices of the chunk

            for idx, label in zip(chunk_indices, chunk):
                label = label.decode("utf-8")
                if allocated[label]["train"] < split_sizes[label]["train"]:
                    split_indices["train"].append(idx)
                    allocated[label]["train"] += 1
                elif allocated[label]["val"] < split_sizes[label]["val"]:
                    split_indices["val"].append(idx)
                    allocated[label]["val"] += 1
                else:
                    split_indices["test"].append(idx)
                    allocated[label]["test"] += 1

    print(
        f"Train size: {len(split_indices['train'])}, Validation size: {len(split_indices['val'])}, Test size: {len(split_indices['test'])} "
        f"& total: {len(split_indices['train']) + len(split_indices['val']) + len(split_indices['test'])}")
    print(f"Loaded {unique_classes}, {len(unique_classes)} classes total")
    print(f"Feature dimension: {feature_dimension}")

    # fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)

    train_batches = math.ceil(len(split_indices['train']) / batch_size)
    val_batches = math.ceil(len(split_indices['val']) / batch_size)
    test_batches = math.ceil(len(split_indices['test']) / batch_size)

    print(f"Train batches: {train_batches}, Validation batches: {val_batches}, Test batches: {test_batches}")

    # Create separate datasets for train, val, and test
    train_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices['train'], batch_size, label_encoder
    ).shuffle(buffer_size=1024).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices['val'], batch_size, label_encoder
    ).repeat().prefetch(tf.data.AUTOTUNE)

    test_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices['test'], batch_size, label_encoder
    ).prefetch(tf.data.AUTOTUNE)

    # Train and evaluate model
    train_and_evaluate_model(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=len(unique_classes),
        save_folder=iteration_save_folder,
        iteration=args.iteration,
        walk_distance=args.walk_distance,
        amount_of_walks=args.amount_of_walks,
        label_encoder=label_encoder
    )
