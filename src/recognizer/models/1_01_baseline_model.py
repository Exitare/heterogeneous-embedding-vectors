import logging
from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, recall_score, precision_score, \
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, mean_absolute_error, mean_squared_error, \
    root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

save_path = Path("results", "recognizer", "baseline")
load_path = Path("results", "recognizer", "summed_embeddings", "multi")

embeddings = ['Text', 'Image', 'RNA', 'Mutation']


def compute_multiclass_confusion_matrices(y_test, y_pred):
    """Computes confusion matrices separately for each output in a multi-output setting."""

    conf_matrices = {}

    for i, modality in enumerate(label_keys):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        conf_matrices[f"Modality: {modality}"] = cm

    return conf_matrices


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

    # assert indices do not overlap
    assert len(set(train_indices).intersection(val_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(val_indices).intersection(test_indices)) == 0
    return train_indices, val_indices, test_indices


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=64, help='The batch size to train the model')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number for the walk distance to work with.', choices=list(range(3, 101)) + [-1])
    parser.add_argument("--run_iteration", "-ri", type=int, required=False, default=1,
                        help="The iteration number for the run. Used for saving the results and validation.")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, required=True,
                        help="The size of the generated summed embeddings count.")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0,
                        help="Ratio of random noise added to the sum embeddings")
    parser.add_argument("--cancer", "-c", nargs="+", required=False,
                        help="The cancer types to work with", default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--epochs", "-e", type=int, default=1000, help="The number of epochs to train the model.")
    parser.add_argument("--multi", action="store_true", help="Create fine grained cancer predictions")
    args = parser.parse_args()

    batch_size: int = args.batch_size
    walk_distance: int = args.walk_distance
    run_iteration: int = args.run_iteration
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    noise_ratio: float = args.noise_ratio
    cancers: [str] = args.cancer
    epochs: int = args.epochs
    multi: bool = args.multi

    if len(cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        cancers = cancers[0].split(" ")

    selected_cancers = "_".join(cancers)

    logging.info(f"Running filename: {__file__}")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Total walk distance: {walk_distance}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run iteration: {run_iteration}")
    logging.info(f"Summed embedding count: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Multi: {multi}")

    load_path: Path = Path(load_path, selected_cancers)

    run_name: str = f"run_{run_iteration}"

    save_path = Path(save_path, "multi" if multi else "simple")

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
        # Explicitly sort keys for consistency
        label_keys = sorted(embeddings) if not multi else sorted(f.keys() - {"X", "meta_information", "WalkDistances"})

        logging.info(f"Label keys: {label_keys}")
        assert len(label_keys) == 4 if not multi else len(
            label_keys) == 10, "Error: Expected 4 modalities for single cancer, or more for multi-cancer."

    logging.info(f"Loaded HDF5 file with {num_samples} samples and input dimension {input_dim}.")

    if walk_distance != -1:
        max_embedding = walk_distance
    else:
        with h5py.File(train_file, "r") as f:
            max_embedding = f["meta_information"].attrs["max_embedding"]
            logging.info(f"Max embedding: {max_embedding}")

    logging.info("Loading data....")

    # Create train-test split indices
    train_indices, val_indices, test_indices = create_indices(train_file, walk_distance)

    with h5py.File(train_file, "r") as f:
        X_train = f["X"][np.sort(train_indices)]
        X_val = f["X"][np.sort(val_indices)]
        X_test = f["X"][np.sort(test_indices)]

        y_train = np.array([f[key][np.sort(train_indices)] for key in label_keys]).T
        y_val = np.array([f[key][np.sort(val_indices)] for key in label_keys]).T
        y_test = np.array([f[key][np.sort(test_indices)] for key in label_keys]).T

    # assert that there is no overlap between X_train, X_val, X_test
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == num_samples

    logging.info("Running model....")
    """Trains separate logistic regression models for each output."""
    model = MultiOutputClassifier(LogisticRegression(max_iter=epochs))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    max_classes = max(estimator.predict_proba(X_test).shape[1] for estimator in model.estimators_)
    # reshape to # (samples, modalities, classes)
    y_pred_proba = np.array([
        np.pad(estimator.predict_proba(X_test), ((0, 0), (0, max_classes - estimator.predict_proba(X_test).shape[1])),
               constant_values=0)  # Pad missing class probabilities with 0
        if estimator.predict_proba(X_test).shape[1] < max_classes else estimator.predict_proba(X_test)
        for estimator in model.estimators_
    ])

    y_pred_proba = np.transpose(y_pred_proba, (1, 0, 2))  # Ensure shape (N, modalities, C)

    # find walk distance for each y_pred
    if walk_distance == -1:
        hdf5_file_path = train_file if noise_ratio == 0.0 else test_file

        if walk_distance == -1:
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                walk_distances = hdf5_file["WalkDistances"][np.sort(test_indices)]  # Ensure sorted indices

    predictions = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,  #
        "walk_distance": walk_distances if walk_distance == -1 else np.repeat(walk_distance, len(y_test))
    }

    # Compute accuracy separately for zero and non-zero values per modality
    metrics = []

    walk_distances = predictions["walk_distance"]
    # get all unique values from numpy array walk distances
    walk_distances = np.unique(walk_distances)
    logging.info(f"Walk distances: {walk_distances}")
    for walk_distance in walk_distances:
        for i, modality in enumerate(label_keys):
            walk_indices = np.where(predictions["walk_distance"] == walk_distance)[0]
            logging.info(f"Walk indices count: {len(walk_indices)}")

            if len(walk_indices) == 0:
                logging.info(f"⚠️ No indices found for walk_distance={walk_distance}! This might be the issue.")

            y_test = predictions["y_test"][walk_indices]
            y_pred = predictions["y_pred"][walk_indices]
            y_pred_proba = predictions["y_pred_proba"][walk_indices]

            unique_classes = np.unique(y_test[:, i])  # Find unique class labels
            num_unique_classes = len(unique_classes)

            # Ensure `y_pred_proba` contains only relevant classes
            class_indices = np.isin(range(y_pred_proba.shape[2]), unique_classes)
            y_pred_proba_adjusted = y_pred_proba[:, i, class_indices]  # Filter probability columns

            zero_mask = y_test[:, i] == 0
            non_zero_mask = y_test[:, i] != 0
            if y_test.shape[0] == 0:
                logging.warning(f"y_test has no samples for modality {modality}!")

            assert np.all(
                np.logical_not(np.logical_and(zero_mask, non_zero_mask))), "Error: zero_mask and non_zero_mask overlap!"

            y_test_non_zero = y_test[non_zero_mask, i]
            y_pred_non_zero = y_pred[non_zero_mask, i]

            y_test_zero = y_test[zero_mask, i]
            y_pred_zero = y_pred[zero_mask, i]

            y_pred_proba_adjusted = y_pred_proba[:, i, :num_unique_classes]
            y_pred_proba_adjusted /= y_pred_proba_adjusted.sum(axis=1, keepdims=True)



            logging.info(f"Found {num_unique_classes} unique classes for modality {modality}.")

            modality_metrics = {}
            modality_metrics["modality"] = modality
            modality_metrics["walk_distance"] = walk_distance
            modality_metrics["balanced_accuracy"] = balanced_accuracy_score(y_test[:, i], y_pred[:, i])

            if modality in embeddings:
                modality_metrics["mcc"] = matthews_corrcoef(y_test[:, i], y_pred[:, i])

                # Get number of classes actually predicted
                num_pred_classes = y_pred_proba_adjusted.shape[1]

                # If a class is missing from predictions, add zero-probability columns
                if num_pred_classes < len(unique_classes):
                    missing_classes = set(unique_classes) - set(range(num_pred_classes))
                    print(f"⚠️ Missing classes in predictions: {missing_classes}")

                    # Pad missing classes with zero probabilities
                    y_pred_proba_padded = np.zeros((y_pred_proba_adjusted.shape[0], len(unique_classes)))
                    y_pred_proba_padded[:, :num_pred_classes] = y_pred_proba_adjusted  # Fill known classes

                    y_pred_proba_adjusted = y_pred_proba_padded  # Replace with padded version

                modality_metrics["auc"] = roc_auc_score(
                    y_test[:, i], y_pred_proba_adjusted, multi_class="ovr", labels=unique_classes
                )
            else:
                unique_classes = np.unique(y_test_non_zero)  # Find unique class labels
                y_pred_proba_adjusted_non_zero = y_pred_proba_adjusted[non_zero_mask]
                num_unique_classes = len(unique_classes)
                # Identify relevant classes (excluding class 0)
                relevant_classes = unique_classes[unique_classes != 0]  # Exclude class 0

                # Identify corresponding indices in probability matrix
                class_indices = np.isin(range(y_pred_proba_adjusted_non_zero.shape[1]), relevant_classes)

                # Select only relevant class probabilities
                y_pred_proba_adjusted_non_zero = y_pred_proba_adjusted_non_zero[:, class_indices]
                # Convert y_test to one-hot encoding
                y_test_non_zero_one_hot = label_binarize(y_test_non_zero, classes=unique_classes)
                modality_metrics["mcc"] = matthews_corrcoef(y_test_non_zero, y_pred_non_zero)
                modality_metrics["auc"] = roc_auc_score(
                    y_test_non_zero_one_hot, y_pred_proba_adjusted_non_zero, multi_class="ovr", labels=unique_classes
                )

            if modality in embeddings:
                # Compute accuracy for zero values
                if np.any(zero_mask):  # Ensure at least one zero entry exists
                    acc_zero = accuracy_score(y_test_zero, y_pred_zero)
                    f1_zero = f1_score(y_test_zero, y_pred_zero, average='weighted')
                    jaccard_zero = jaccard_score(y_test_zero, y_pred_zero, average='weighted')
                    recall_zero = recall_score(y_test_zero, y_pred_zero, average='weighted')
                    precision_zero = precision_score(y_test_zero, y_pred_zero, average='weighted')
                    mae_zero = mean_absolute_error(y_test_zero, y_pred_zero)
                    rmse_zero = root_mean_squared_error(y_test_zero, y_pred_zero)
                    mse_zero = mean_squared_error(y_test_zero, y_pred_zero)

                    modality_metrics["accuracy_zero"] = acc_zero
                    modality_metrics["f1_zero"] = f1_zero
                    modality_metrics["recall_zero"] = recall_zero
                    modality_metrics["precision_zero"] = precision_zero
                    modality_metrics["mae_zero"] = mae_zero
                    modality_metrics["rmse_zero"] = rmse_zero
                    modality_metrics["mse_zero"] = mse_zero

            # Compute accuracy for non-zero values
            if np.any(non_zero_mask):  # Ensure at least one non-zero entry exists
                acc_non_zero = accuracy_score(y_test_non_zero, y_pred_non_zero)
                f1_non_zero = f1_score(y_test_non_zero, y_pred_non_zero, average='weighted')
                recall_non_zero = recall_score(y_test_non_zero, y_pred_non_zero, average='weighted')
                precision_non_zero = precision_score(y_test_non_zero, y_pred_non_zero, average='weighted')
                mae_non_zero = mean_absolute_error(y_test_non_zero, y_pred_non_zero)
                rmse_non_zero = root_mean_squared_error(y_test_non_zero, y_pred_non_zero)
                mse_non_zero = mean_squared_error(y_test_non_zero, y_pred_non_zero)

                modality_metrics["accuracy"] = acc_non_zero
                modality_metrics["f1"] = f1_non_zero
                modality_metrics["recall"] = recall_non_zero
                modality_metrics["precision"] = precision_non_zero
                modality_metrics["mae"] = mae_non_zero
                modality_metrics["rmse"] = rmse_non_zero
                modality_metrics["mse"] = mse_non_zero

            metrics.append(modality_metrics)

    # create confusion matrices
    # Compute confusion matrices for each output
    conf_matrices = compute_multiclass_confusion_matrices(y_test, y_pred)

    # Print the confusion matrix for each output
    for output, matrix in conf_matrices.items():
        print(f"{output}:\n{matrix}\n")

    metrics_df = pd.DataFrame(metrics)

    print(metrics_df)

    # save the metrics
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)
    logging.info(f"Saved metrics to {Path(save_path, 'metrics.csv')}")
