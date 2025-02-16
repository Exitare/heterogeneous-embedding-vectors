from argparse import ArgumentParser
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, mean_absolute_error, mean_squared_error, \
    root_mean_squared_error, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_folder = Path("results", "recognizer")

base_embeddings = ["RNA", "Mutation", "Image", "Text"]

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--model", "-m", choices=["multi", "simple"], default="multi",
                        help="The model to use")

    args = parser.parse_args()
    model: str = args.model
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    selected_cancers: [str] = '_'.join(cancers)

    logging.info(
        f"Loading data for model: {model}, cancers: {cancers}, amount_of_walk_embeddings: {amount_of_walk_embeddings}")

    load_folder = Path(load_folder, model, selected_cancers, str(amount_of_walk_embeddings))
    logging.info(f"Loading data from {load_folder}...")

    processed_files = []
    for noise_folder in load_folder.iterdir():
        if noise_folder.is_file():
            continue

        for walk_distance_folder in noise_folder.iterdir():
            if walk_distance_folder.is_file():
                continue

            for run_folder in walk_distance_folder.iterdir():
                if run_folder.is_file():
                    continue

                noise_ratio = float(run_folder.parts[-3])
                print(run_folder)

                # Reset the AUC dictionary for this run folder
                aucs = {}
                results = []

                # Process probabilities.json files for this run folder
                for file in run_folder.iterdir():
                    if file.is_file() and "probabilities.json" in file.parts:
                        df = pd.read_json(file, orient="records", lines=True)
                        for walk_distance in df["walk_distance"].unique():
                            walk_distance_df = df[df["walk_distance"] == walk_distance]
                            for embedding in walk_distance_df["embedding"].unique():
                                embedding_data = walk_distance_df[walk_distance_df["embedding"] == embedding]
                                y_true = embedding_data["y_true"]
                                y_prob_series = embedding_data["y_prob"]
                                # Convert the y_prob column (each element is a list) to a 2D NumPy array
                                y_prob_array = np.array(y_prob_series.tolist())
                                # logging.info(
                                #    f"Processing probabilities for walk_distance {walk_distance}, embedding {embedding}")
                                # logging.info(f"Unique labels in y_true: {y_true.unique()}")

                                # Determine present labels (assumes model output columns correspond to label values)
                                present_labels = sorted(y_true.unique())
                                # Slice the full probability array to only include the columns for present labels,
                                # then renormalize so each row sums to 1.
                                y_prob_array_adjusted = y_prob_array[:, present_labels]
                                y_prob_array_adjusted = y_prob_array_adjusted / np.sum(y_prob_array_adjusted, axis=1,
                                                                                       keepdims=True)

                                try:
                                    auc = roc_auc_score(y_true, y_prob_array_adjusted, multi_class="ovr",
                                                        labels=present_labels)
                                except Exception as e:
                                    logging.error(
                                        f"Error computing ROC AUC for walk_distance {walk_distance}, embedding {embedding}: {e}")
                                    auc = np.nan
                                # Create a consistent key (make sure walk_distance is an integer, embedding is a string)
                                key = (int(walk_distance), str(embedding))
                                aucs[key] = auc
                                # logging.info(f"Computed AUC for {key}: {auc}")

                # Now process predictions.csv files for this run folder and add the corresponding AUCs:
                for file in run_folder.iterdir():
                    if file.is_file() and "predictions.csv" in file.parts:
                        df = pd.read_csv(file)
                        for walk_distance in df["walk_distance"].unique():
                            walk_distance_df = df[df["walk_distance"] == walk_distance]
                            for embedding in walk_distance_df["embedding"].unique():
                                if embedding in base_embeddings:
                                    embedding_data = walk_distance_df[walk_distance_df["embedding"] == embedding]
                                    y_true = embedding_data["y_true"]
                                    y_pred = embedding_data["y_pred"]

                                    y_true_non_zero = y_true[y_true != 0]
                                    y_pred_non_zero = y_pred[y_true != 0]
                                    y_true_zero = y_true[y_true == 0]
                                    y_pred_zero = y_pred[y_true == 0]

                                    labels_to_use = sorted(list(set(y_true_non_zero) | set(y_pred_non_zero)))
                                    if len(y_true_zero) != 0:
                                        mae_zero = mean_absolute_error(y_true_zero, y_pred_zero)
                                        mse_zero = mean_squared_error(y_true_zero, y_pred_zero)
                                        rmse_zero = root_mean_squared_error(y_true_zero, y_pred_zero)
                                        f1_zero = f1_score(y_true_zero, y_pred_zero, average="weighted",
                                                           zero_division=0)
                                        precision_zero = precision_score(y_true_zero, y_pred_zero, average="weighted",
                                                                         zero_division=0)
                                        recall_zero = recall_score(y_true_zero, y_pred_zero, average="weighted",
                                                                   zero_division=0)
                                        accuracy_zero = accuracy_score(y_true_zero, y_pred_zero)
                                    else:
                                        mae_zero = np.nan
                                        mse_zero = np.nan
                                        rmse_zero = np.nan
                                        f1_zero = np.nan
                                        precision_zero = np.nan
                                        recall_zero = np.nan
                                        accuracy_zero = np.nan

                                    if len(y_true_non_zero) != 0:
                                        mcc = matthews_corrcoef(y_true, y_pred)
                                        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                                        mae = mean_absolute_error(y_true_non_zero, y_pred_non_zero)
                                        mse = mean_squared_error(y_true_non_zero, y_pred_non_zero)
                                        rmse = root_mean_squared_error(y_true_non_zero, y_pred_non_zero)
                                        f1 = f1_score(y_true_non_zero, y_pred_non_zero, average="weighted",
                                                      labels=labels_to_use, zero_division=0)
                                        precision = precision_score(y_true_non_zero, y_pred_non_zero,
                                                                    average="weighted", zero_division=0)
                                        recall = recall_score(y_true_non_zero, y_pred_non_zero, average="weighted",
                                                              zero_division=0)
                                        specificity = specificity_score(y_true_non_zero, y_pred_non_zero,
                                                                        average="weighted")
                                        sensitivity = sensitivity_score(y_true_non_zero, y_pred_non_zero,
                                                                        average="weighted")
                                        accuracy = accuracy_score(y_true_non_zero, y_pred_non_zero)
                                    else:
                                        mcc = np.nan
                                        balanced_accuracy = np.nan
                                        mae = np.nan
                                        mse = np.nan
                                        rmse = np.nan
                                        f1 = np.nan
                                        precision = np.nan
                                        recall = np.nan
                                        specificity = np.nan
                                        sensitivity = np.nan
                                        accuracy = np.nan

                                    key = (int(walk_distance), str(embedding))
                                    results.append({
                                        "walk_distance": walk_distance,
                                        "embedding": embedding,
                                        "noise": noise_ratio,
                                        "mcc": mcc,
                                        "balanced_accuracy": balanced_accuracy,
                                        "mae": mae,
                                        "mse": mse,
                                        "rmse": rmse,
                                        "mae_zero": mae_zero,
                                        "mse_zero": mse_zero,
                                        "rmse_zero": rmse_zero,
                                        "f1": f1,
                                        "specificity": specificity,
                                        "sensitivity": sensitivity,
                                        "accuracy": accuracy,
                                        "accuracy_zero": accuracy_zero,
                                        "auc": aucs.get(key, np.nan),
                                    })

                                else:
                                    # Handle embeddings not in base_embeddings similarly...
                                    embedding_data = walk_distance_df[walk_distance_df["embedding"] == embedding]
                                    embedding_data_filtered = embedding_data[
                                        (embedding_data["y_true"] != 0) & (embedding_data["y_pred"] != 0)]
                                    y_true_non_zero = embedding_data_filtered["y_true"]
                                    y_pred_non_zero = embedding_data_filtered["y_pred"]

                                    if len(y_true_non_zero) != 0:
                                        mcc = matthews_corrcoef(y_true_non_zero, y_pred_non_zero)
                                        balanced_accuracy = balanced_accuracy_score(y_true_non_zero, y_pred_non_zero)
                                        mae = mean_absolute_error(y_true_non_zero, y_pred_non_zero)
                                        mse = mean_squared_error(y_true_non_zero, y_pred_non_zero)
                                        rmse = root_mean_squared_error(y_true_non_zero, y_pred_non_zero)
                                        f1 = f1_score(y_true_non_zero, y_pred_non_zero, average="weighted",
                                                      zero_division=0)
                                        specificity = specificity_score(y_true_non_zero, y_pred_non_zero,
                                                                        average="weighted")
                                        sensitivity = sensitivity_score(y_true_non_zero, y_pred_non_zero,
                                                                        average="weighted")
                                        accuracy = accuracy_score(y_true_non_zero, y_pred_non_zero)
                                    else:
                                        mcc = np.nan
                                        balanced_accuracy = np.nan
                                        mae = np.nan
                                        mse = np.nan
                                        rmse = np.nan
                                        f1 = np.nan
                                        specificity = np.nan
                                        sensitivity = np.nan
                                        accuracy = np.nan

                                    key = (int(walk_distance), str(embedding))
                                    results.append({
                                        "walk_distance": walk_distance,
                                        "embedding": embedding,
                                        "noise": noise_ratio,
                                        "mcc": mcc,
                                        "balanced_accuracy": balanced_accuracy,
                                        "mae": mae,
                                        "mse": mse,
                                        "rmse": rmse,
                                        "f1": f1,
                                        "specificity": specificity,
                                        "sensitivity": sensitivity,
                                        "accuracy": accuracy,
                                        "auc": aucs.get(key, np.nan),
                                    })

                            df_results = pd.DataFrame(results)
                            df_results.to_csv(Path(run_folder, "metrics.csv"), index=False)
                            processed_files.append(file)

    logging.info(f"Processed {len(processed_files)} files.")
