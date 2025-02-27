import h5py
import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import math
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_model(input_dim, num_classes):
    """Builds and compiles the model."""
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    x = BatchNormalization()(input_layer)
    # First dense block with L2 regularization and dropout
    x = tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Second dense block with L2 regularization and dropout
    #x = tf.keras.layers.Dense(128, activation='relu',
    #                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_cancer_names():

    with open("data/cancer2name.json") as file:
        cancer_names = json.load(file)
    return cancer_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cancer", "-c", nargs="+", default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"],
                        help="Cancer types to work with.")
    parser.add_argument("--iteration", "-i", type=int, required=True, help="Iteration number.")
    parser.add_argument("--walk_distance", "-w", type=int, choices=[3, 4, 5, 6], default=3, help="Walk distance.")
    parser.add_argument("--amount_of_walks", "-a", type=int, choices=[3, 4, 5, 6], default=3, help="Amount of walks.")
    args = parser.parse_args()

    cancer_names = load_cancer_names()
    batch_size = args.batch_size
    walk_distance = args.walk_distance
    walk_amount = args.amount_of_walks
    iteration = args.iteration
    cancers = "_".join(args.cancer)

    # Set paths for loading and saving results
    base_load_folder = Path("results", "sub_type_classifier", "summed_embeddings")
    base_save_folder = Path("results", "sub_type_classifier", "classification")
    load_folder = Path(base_load_folder, cancers, f"{walk_distance}_{walk_amount}")
    h5_file_path = Path(load_folder, "summed_embeddings.h5")
    cancer_save_folder = Path(base_save_folder, cancers, f"{walk_distance}_{walk_amount}")
    iteration_save_folder = Path(cancer_save_folder, str(iteration))
    cancer_save_folder.mkdir(parents=True, exist_ok=True)
    iteration_save_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from: {h5_file_path}")
    logging.info(f"Saving results to: {iteration_save_folder}")

    # Set train/test split ratios
    train_ratio = 0.7

    # Load entire dataset from HDF5 into memory
    with h5py.File(h5_file_path, "r") as h5_file:
        feature_dimension = h5_file.attrs["feature_shape"]
        X = h5_file["X"][:]  # shape (n_samples, feature_dimension)
        y = np.array([label.decode("utf-8") for label in h5_file["y"][:]])

        # For each key (target label) and its list of subtypes,
        # replace every occurrence of each subtype in filtered_labels with the key.
        # Iterate over the indices so that we can update filtered_labels in place
    for i, label in enumerate(y):
        # For each main cancer type, check its subtypes
        for main, subtype_mapping in cancer_names.items():
            if label in subtype_mapping:
                new_val = subtype_mapping[label]
                try:
                    new_val = int(new_val[0])
                except ValueError:
                    new_val = new_val
                    y[i] = new_val

                # Once replaced, we can break out of the inner loop
                break

    # Compute class counts and determine split sizes per class
    class_counts = Counter(y)
    split_sizes = {
        cls: {
            "train": int(count * train_ratio),
            "test": count - int(count * train_ratio)
        }
        for cls, count in class_counts.items()
    }
    # Remove classes with fewer than 30 training samples
    for key in list(split_sizes.keys()):
        if split_sizes[key]["train"] < 150:
            del split_sizes[key]

    logging.info(f"Split sizes per class: {split_sizes}")

    # Get valid indices (only indices of labels in our selected classes)
    valid_indices = [i for i, label in enumerate(y) if label in split_sizes]
    filtered_labels = np.array([y[i] for i in valid_indices])
    unique_filtered_classes = np.unique(filtered_labels)



    # Build split indices (train and test) while respecting per-class counts
    split_indices = {"train": [], "test": []}
    allocated = defaultdict(lambda: {"train": 0, "test": 0})
    for idx in valid_indices:
        label = y[idx]
        if allocated[label]["train"] < split_sizes[label]["train"]:
            split_indices["train"].append(idx)
            allocated[label]["train"] += 1
        else:
            split_indices["test"].append(idx)
            allocated[label]["test"] += 1

    logging.info(f"Train indices count: {len(split_indices['train'])}, Test indices count: {len(split_indices['test'])}")
    logging.info(f"Loaded classes: {unique_filtered_classes}, Total classes: {len(unique_filtered_classes)}")
    logging.info(f"Feature dimension: {feature_dimension}")

    # Create in-memory train and test sets using the indices
    X_train = X[split_indices["train"]]
    y_train = np.array([y[i] for i in split_indices["train"]])
    X_test = X[split_indices["test"]]
    y_test = np.array([y[i] for i in split_indices["test"]])

    # Fit a label encoder on the filtered classes and transform the labels
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_filtered_classes)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    logging.info(f"Training data shape: {X_train.shape}, {y_train_enc.shape}")
    logging.info(f"Test data shape: {X_test.shape}, {y_test_enc.shape}")

    num_classes = len(unique_filtered_classes)
    model = build_model(input_dim=feature_dimension, num_classes=num_classes)

    # Set up callbacks (e.g., early stopping on validation loss)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    ]

    # Train the model using the entire training data in memory
    history = model.fit(
        X_train, y_train_enc,
        epochs=50,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks
    )

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test_enc, batch_size=batch_size)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Generate predictions on the test set
    y_pred_proba = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred_proba, axis=1)

    predictions_df = pd.DataFrame({
        "y_test": y_test_enc,
        "y_test_decoded": label_encoder.inverse_transform(y_test_enc),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred)
    })
    predictions_df.to_csv(Path(iteration_save_folder, "predictions.csv"), index=False)

    # Calculate per-class metrics
    results = []
    for class_index in np.unique(y_test_enc):
        mask = (y_test_enc == class_index)
        y_test_class = y_test_enc[mask]
        y_pred_class = y_pred[mask]
        class_name = label_encoder.inverse_transform([class_index])[0]
        accuracy_class = np.mean(y_test_class == y_pred_class)
        f1_class = f1_score(y_test_class, y_pred_class, average='weighted')
        precision_class = precision_score(y_test_class, y_pred_class, average='weighted')
        recall_class = recall_score(y_test_class, y_pred_class, average='weighted')
        logging.info(f"{class_name}: Accuracy: {accuracy_class:.4f}, F1: {f1_class:.4f}, Precision: {precision_class:.4f}, Recall: {recall_class:.4f}")
        results.append({
            "cancer": class_name,
            "accuracy": accuracy_class,
            "f1": f1_class,
            "precision": precision_class,
            "recall": recall_class,
            "iteration": iteration,
            "walk_distance": walk_distance,
            "amount_of_walks": walk_amount
        })

    # Overall metrics
    overall_accuracy = np.mean(y_test_enc == y_pred)
    overall_f1 = f1_score(y_test_enc, y_pred, average='weighted')
    overall_precision = precision_score(y_test_enc, y_pred, average='weighted')
    overall_recall = recall_score(y_test_enc, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test_enc, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_enc, y_pred)
    # Compute AUC (requires one-hot encoding of y_test)

    if len(unique_filtered_classes) != 2:
        y_test_one_hot = label_binarize(y_test_enc, classes=np.arange(num_classes))
        auc_score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovo', average='macro')
    else:
        auc_score = roc_auc_score(y_test_enc, y_pred_proba[:, 1])

    logging.info(f"Overall: Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}, Precision: {overall_precision:.4f}, "
                 f"Recall: {overall_recall:.4f}, AUC: {auc_score:.4f}, MCC: {mcc:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
    results.append({
        "cancer": "All",
        "accuracy": overall_accuracy,
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "auc": auc_score,
        "iteration": iteration,
        "walk_distance": walk_distance,
        "amount_of_walks": walk_amount
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(iteration_save_folder, "results.csv"), index=False)

    model.save(Path(iteration_save_folder, "model.keras"))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(iteration_save_folder, "history.csv"), index=False)
    logging.info("Training, evaluation, and saving complete.")


if __name__ == "__main__":
    main()