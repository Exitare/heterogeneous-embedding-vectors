import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIG_ROOT = Path("figures", "single_modality_classifier")
CLASS_ROOT = Path("results", "single_modality_classifier", "classification")

def main():
    parser = argparse.ArgumentParser(description="Aggregate predictions and plot confusion matrix")
    parser.add_argument("--cancer", "-c", nargs="+", required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"],
                        help="Cancer types to work with.")
    parser.add_argument("--selected_modality", "-sm", type=str, required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modality used for classification (matches training output).")
    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers: str = "_".join(selected_cancers)
    modality: str = args.selected_modality

    load_path: Path = Path(CLASS_ROOT,cancers, modality)
    if not load_path.exists():
        raise FileNotFoundError(f"Classification folder not found: {load_path}")

    all_predictions = []
    for run_dir in sorted(load_path.iterdir()):
        if not run_dir.is_dir():
            continue
        pred_csv = run_dir / "predictions.csv"
        if pred_csv.exists():
            logging.info(f"Loading {pred_csv}")
            all_predictions.append(pd.read_csv(pred_csv))

    if not all_predictions:
        raise FileNotFoundError(f"No predictions.csv found under {load_path}/*/")

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Build confusion matrix (use sorted labels for stable axes)
    true_labels = predictions["y_test_decoded"].astype(str)
    pred_labels = predictions["y_pred_decoded"].astype(str)
    labels = sorted(set(true_labels) | set(pred_labels))

    confusion_matrix = pd.crosstab(true_labels, pred_labels,
                                   rownames=["True"], colnames=["Predicted"]).reindex(index=labels, columns=labels, fill_value=0)

    # Save figure
    fig_save_folder: Path = Path(FIG_ROOT, cancers, modality, "performance")
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.title(f"Confusion Matrix • {modality.capitalize()} • {' '.join(cancers.split('_'))}")
    plt.tight_layout()
    out_path = fig_save_folder / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix to: {out_path}")

if __name__ == "__main__":
    main()