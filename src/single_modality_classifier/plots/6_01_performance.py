#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Roots
FIG_ROOT = Path("figures", "single_modality_classifier")
CLASS_ROOT = Path("results", "single_modality_classifier", "classification")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "Matthews Correlation Coefficient",
    "balanced_accuracy": "Balanced Accuracy",
    "auc": "AUC",
}

COLOR_PALETTE = {
    "Annotation": "#c8b7b7ff",
    "Image": "#d38d5fff",
    "RNA": "#c6afe9ff",
    "Mutation": "#de87aaff",
    "BRCA": "#c837abff",
    "LUAD": "#37abc8ff",
    "BLCA": "#ffcc00ff",
    "THCA": "#d35f5fff",
    "STAD": "#f47e44d7",
    "COAD": "#502d16ff",
    "All": "#000000",
}

def load_all_results(cancers: str, modality: str) -> pd.DataFrame:
    """Load results.csv from each iteration folder."""
    base = CLASS_ROOT / cancers / modality
    if not base.exists():
        raise FileNotFoundError(f"Classification folder not found: {base}")
    dfs = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            logging.info(f"Loading {csv_path}")
            df = pd.read_csv(csv_path)
            # attach iteration id from folder name
            try:
                iter_id = int(run_dir.name)
            except ValueError:
                iter_id = run_dir.name
            df["iteration"] = iter_id
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No results.csv files found under {base}/*/")
    out = pd.concat(dfs, ignore_index=True)
    out["cancer"] = out["cancer"].astype(str)
    return out

def create_hue_performance_plot(df: pd.DataFrame, metric: str, save_folder: Path):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="cancer", y=metric, hue="cancer", palette=COLOR_PALETTE)
    plt.title(f"{METRIC_LABELS.get(metric, metric)}")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(save_folder / f"{metric}_score.png", dpi=300)
    plt.close('all')

def create_performance_overview_plot(df: pd.DataFrame, save_folder: Path):
    sub = df[df["cancer"] == "All"].copy()
    if sub.empty:
        logging.warning("No 'All' rows found; skipping overview plot.")
        return
    melted = sub.melt(id_vars=["cancer"], value_vars=["precision", "recall", "auc"],
                      var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].replace({"precision": "Precision", "recall": "Recall", "auc": "AUC"})
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="cancer", y="score", hue="metric")
    plt.title("Overview scores")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(save_folder / "overview_scores.png", dpi=300)
    plt.close('all')

def create_overall_trends_by_iteration(df: pd.DataFrame, save_folder: Path):
    sub = df[df["cancer"] == "All"].copy()
    if sub.empty:
        logging.warning("No 'All' rows found; skipping trends plot.")
        return
    melted = sub.melt(id_vars=["iteration"],
                      value_vars=["precision", "recall", "auc", "accuracy", "f1", "mcc", "balanced_accuracy"],
                      var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].replace({k: METRIC_LABELS.get(k, k) for k in melted["metric"].unique()})
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=melted, x="iteration", y="score", hue="metric", marker="o")
    plt.title("Metrics over iterations (All)")
    plt.ylabel("Score")
    plt.xlabel("Iteration")
    plt.ylim(0, 1)
    plt.legend(title="", ncols=3, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_folder / "overview_scores_by_iteration.png", dpi=300)
    plt.close('all')

def create_metric_heatmap_by_cancer(df: pd.DataFrame, save_folder: Path):
    keep = ["precision", "recall", "auc", "accuracy", "f1", "mcc", "balanced_accuracy"]
    agg = df.groupby("cancer", as_index=True)[keep].mean()
    agg = agg.rename(columns={k: METRIC_LABELS.get(k, k) for k in keep})
    plt.figure(figsize=(10, 6))
    sns.heatmap(agg, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5, vmin=0.0, vmax=1.0)
    plt.title("Mean Metrics by Cancer (across iterations)")
    plt.xlabel("Metric")
    plt.ylabel("Cancer")
    plt.tight_layout()
    plt.savefig(save_folder / "metrics_heatmap_by_cancer.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def create_box_plot_by_cancer(df: pd.DataFrame, metric: str, save_folder: Path):
    plt.figure(figsize=(10, 5))
    sns.boxenplot(data=df, x="cancer", y=metric, palette=COLOR_PALETTE)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} by Cancer (across iterations)")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.xlabel("")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_folder / f"{metric}_by_cancer_box.png", dpi=300)
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarize single_modality_classifier results")
    parser.add_argument("--cancer", "-c", nargs="+", required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"],
                        help="Cancer types to work with.")
    parser.add_argument("--selected_modality", "-sm", type=str, required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modality used for classification.")
    args = parser.parse_args()

    cancers_list = args.cancer
    cancers = "_".join(cancers_list)
    modality = args.selected_modality

    logging.info(f"Using cancer types: {cancers_list}")
    logging.info(f"Selected modality: {modality}")

    # Save folder
    save_folder = FIG_ROOT / cancers / modality / "performance"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_all_results(cancers, modality)
    results.reset_index(drop=True, inplace=True)

    # Plots
    create_hue_performance_plot(results, "accuracy", save_folder)
    create_hue_performance_plot(results, "f1", save_folder)
    create_performance_overview_plot(results, save_folder)
    create_overall_trends_by_iteration(results, save_folder)
    create_metric_heatmap_by_cancer(results, save_folder)
    create_box_plot_by_cancer(results, "accuracy", save_folder)
    create_box_plot_by_cancer(results, "f1", save_folder)

    logging.info(f"Saved figures to: {save_folder}")