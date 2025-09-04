import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="The cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--walk_distance", "-w", type=int, required=True,
                        help="The walk distance used for the classification.")
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True,
                        help="The amount of walks used for the classification.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)
    walk_distance = args.walk_distance
    amount_of_walks = args.amount_of_walks

    all_predictions = []

    load_path = Path("results", "classifier_modality_adjusted", "classification", cancers,
                     f"{walk_distance}_{amount_of_walks}")
    for run_directory in load_path.iterdir():
        if run_directory.is_file():
            continue

        path: Path
        for path in run_directory.iterdir():
            if path.is_dir():
                continue

            if "predictions.csv" in str(path):
                logging.info(f"Loading {path}")
                predictions = pd.read_csv(path)
                all_predictions.append(predictions)

    predictions = pd.concat(all_predictions)

    # create confusion matrix
    confusion_matrix = pd.crosstab(predictions["y_test_decoded"], predictions["y_pred_decoded"], rownames=['True'],
                                   colnames=['Predicted'])

    # visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g')
    save_folder: Path = Path("figures", "classifier_modality_adjusted", cancers, "performance")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)
    plt.savefig(Path(save_folder, f"{walk_distance}_{amount_of_walks}_confusion_matrix.png"),
                dpi=300)
