import h5py
import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, \
    balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import math
import logging

# I/O folders
load_folder_root = Path("results", "single_modality_classifier", "summed_embeddings")
save_folder_root = Path("results", "single_modality_classifier", "classification")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def h5_generator_specific_indices(h5_file_path, indices, batch_size, label_encoder):
    """
    Generator that yields batches of data based on specific indices.
    h5py requires fancy-index selections to be in strictly increasing order.
    """
    num_samples = len(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = np.asarray(indices[start_idx:end_idx], dtype=np.int64)

        # h5py: indices must be strictly increasing
        order = np.argsort(batch_indices)
        sorted_idx = batch_indices[order]

        with h5py.File(h5_file_path, "r") as h5_file:
            X_batch = h5_file["X"][sorted_idx]
            y_batch = h5_file["y"][sorted_idx]

        # Decode and encode labels
        y_decoded = [lab.decode("utf-8") if isinstance(lab, bytes) else str(lab) for lab in y_batch]
        y_batch_enc = label_encoder.transform(y_decoded)

        yield X_batch.astype("float32", copy=False), np.asarray(y_batch_enc, dtype="int32")


def create_tf_dataset_specific_indices(h5_file_path, indices, batch_size, label_encoder, repeat=False, shuffle=False):
    """
    Creates a TensorFlow dataset from specific indices.
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        feature_dimension = int(h5_file.attrs["feature_shape"])

    generator = lambda: h5_generator_specific_indices(h5_file_path, indices, batch_size, label_encoder)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, feature_dimension), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def train_and_evaluate_model(train_ds, val_ds, test_ds, num_classes, save_folder, iteration, label_encoder,
                             train_steps, val_steps):
    """
    Simple MLP classifier with class weighting and early stopping.
    """
    input_layer = tf.keras.layers.Input(shape=(train_ds.element_spec[0].shape[1],))
    x = BatchNormalization()(input_layer)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Class weights (optional; only applied if those classes exist)
    classes = list(label_encoder.classes_)
    class_weights = {i: 1.0 for i in range(num_classes)}
    if "LUAD" in classes:
        class_weights[classes.index("LUAD")] = 6.0
    if "BLCA" in classes:
        class_weights[classes.index("BLCA")] = 2.5
    # if "BRCA" in classes: class_weights[classes.index("BRCA")] = 2.0  # example tweak if desired

    history = model.fit(
        train_ds,
        epochs=50,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max",
                                                    restore_best_weights=True)],
        verbose=1,
    )

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Collect full test predictions
    y_test, y_pred, y_pred_proba = [], [], []
    for X_batch, y_batch in test_ds:
        probs = model.predict(X_batch, verbose=0)
        y_pred_proba.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_test.append(y_batch.numpy())
    y_test = np.concatenate(y_test, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)

    # Save predictions (both encoded and decoded)
    predictions = pd.DataFrame({
        "y_test": y_test,
        "y_test_decoded": label_encoder.inverse_transform(y_test),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred),
    })
    predictions.to_csv(Path(save_folder, "predictions.csv"), index=False)

    # Per-class metrics
    results = []
    for cls_idx in np.unique(y_test):
        mask = (y_test == cls_idx)
        yt, yp = y_test[mask], y_pred[mask]
        cancer_name = label_encoder.inverse_transform([cls_idx])[0]
        results.append({
            "cancer": cancer_name,
            "accuracy": (yt == yp).mean(),
            "f1": f1_score(yt, yp, average="weighted"),
            "precision": precision_score(yt, yp, average="weighted"),
            "recall": recall_score(yt, yp, average="weighted"),
            "iteration": iteration,
        })

    # Overall metrics
    f1_total = f1_score(y_test, y_pred, average="weighted")
    precision_total = precision_score(y_test, y_pred, average="weighted")
    recall_total = recall_score(y_test, y_pred, average="weighted")
    accuracy_total = (y_test == y_pred).mean()
    mcc_total = matthews_corrcoef(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # AUC (one-vs-one macro)
    num_classes_inferred = y_pred_proba.shape[1]
    y_test_one_hot = label_binarize(y_test, classes=np.arange(num_classes_inferred))
    try:
        auc_score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class="ovo", average="macro")
    except ValueError:
        # If a class is missing from test set, AUC may be undefined — handle gracefully
        auc_score = float("nan")

    results.append({
        "cancer": "All",
        "accuracy": accuracy_total,
        "f1": f1_total,
        "mcc": mcc_total,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_total,
        "recall": recall_total,
        "auc": auc_score,
        "iteration": iteration,
    })

    # Save results + artifacts
    pd.DataFrame(results).to_csv(Path(save_folder, "results.csv"), index=False)
    logging.info("Results saved.")

    model.save(Path(save_folder, "model.keras"))
    pd.DataFrame(history.history).to_csv(Path(save_folder, "history.csv"), index=False)
    logging.info("Model and history saved.")


def stratified_random_split(y, train_ratio=0.7, val_ratio=0.05, test_ratio=0.25, rng=None):
    """
    Per-class random split returning dict of indices: {'train': [], 'val': [], 'test': []}.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y)
    indices_by_class = defaultdict(list)
    for idx, lab in enumerate(y):
        indices_by_class[lab].append(idx)

    split_indices = {"train": [], "val": [], "test": []}
    for lab, idxs in indices_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        split_indices["train"].extend(idxs[:n_train].tolist())
        split_indices["val"].extend(idxs[n_train:n_train + n_val].tolist())
        split_indices["test"].extend(idxs[n_train + n_val:].tolist())

    # Shuffle each split for good measure
    for k in split_indices:
        rng.shuffle(split_indices[k])

    return split_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="Cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--selected_modality", "-sm", type=str, choices=["rna", "annotations", "mutations", "images"],
                        required=True, help="The single modality to classify from.")
    parser.add_argument("--iteration", "-i", type=int, required=True, help="Iteration number (for saving outputs).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible splits (optional).")
    args = parser.parse_args()

    batch_size = args.batch_size
    iteration = args.iteration
    cancers = "_".join(args.cancer)
    selected_modality = args.selected_modality

    logging.info(f"Selected cancers: {cancers}")
    logging.info(f"Selected modality: {selected_modality}")

    # Paths
    load_folder = Path(load_folder_root, cancers)
    h5_file_path = Path(load_folder, f"{selected_modality}_embeddings.h5")

    cancer_save_folder = Path(save_folder_root, cancers, selected_modality)
    iteration_save_folder = Path(cancer_save_folder, str(iteration))
    iteration_save_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from: {h5_file_path}")
    logging.info(f"Saving results to: {iteration_save_folder}")

    # Load entire dataset
    with h5py.File(h5_file_path, "r") as h5_file:
        feature_dimension = int(h5_file.attrs["feature_shape"])
        # classes stored as utf-8 strings
        unique_classes = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in h5_file.attrs["classes"][:]]
        X = h5_file["X"][:]  # (N, D)
        raw_y = h5_file["y"][:]
        y = np.array([lab.decode("utf-8") if isinstance(lab, bytes) else str(lab) for lab in raw_y])

    logging.info(f"Loaded {len(unique_classes)} classes: {unique_classes}")
    logging.info(f"Feature dimension: {feature_dimension}, Samples: {len(y)}")

    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)

    # Fresh random split every run (seed=None => new each run)
    rng = np.random.default_rng(args.seed)
    split_indices = stratified_random_split(y, train_ratio=0.7, val_ratio=0.05, test_ratio=0.25, rng=rng)

    logging.info(
        f"Train size: {len(split_indices['train'])}, "
        f"Validation size: {len(split_indices['val'])}, "
        f"Test size: {len(split_indices['test'])}, "
        f"Total: {sum(len(v) for v in split_indices.values())}"
    )

    # Sanity checks (all classes present in each split)
    assert set(np.unique(y[split_indices["train"]])) == set(np.unique(y)), "Missing classes in training set!"
    assert set(np.unique(y[split_indices["val"]])) == set(np.unique(y)), "Missing classes in validation set!"
    assert set(np.unique(y[split_indices["test"]])) == set(np.unique(y)), "Missing classes in test set!"

    # Batch/step sizes
    train_batches = math.ceil(len(split_indices["train"]) / batch_size)
    val_batches = math.ceil(len(split_indices["val"]) / batch_size)
    test_batches = math.ceil(len(split_indices["test"]) / batch_size)

    logging.info(f"Train batches: {train_batches}, Val batches: {val_batches}, Test batches: {test_batches}")

    # Build datasets
    train_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices["train"], batch_size, label_encoder, repeat=True, shuffle=True
    )
    val_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices["val"], batch_size, label_encoder, repeat=True, shuffle=False
    )
    test_ds = create_tf_dataset_specific_indices(
        h5_file_path, split_indices["test"], batch_size, label_encoder, repeat=False, shuffle=False
    )

    # Train & evaluate
    train_and_evaluate_model(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=len(unique_classes),
        save_folder=iteration_save_folder,
        iteration=iteration,
        label_encoder=label_encoder,
        train_steps=train_batches,
        val_steps=val_batches,
    )
