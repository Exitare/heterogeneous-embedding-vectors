from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging

save_folder = Path("figures", "classifier")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_grid_plot(df: pd.DataFrame, metric: str):
    # create a grid plot for the accuracy, each unique value of walks should have a separate plot, the hue is cancer
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="walks", y=metric, hue="cancer", ax=ax,
                order=["3_3", "3_4", "3_5", "4_3", "4_4", "4_5", "5_3", "5_4", "5_5"])
    ax.set_title(f"{metric.upper()} Score")
    ax.set_ylabel(f"{metric.upper()} Score")
    ax.set_xlabel("")

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [f"Distance: {label.split('_')[0]}\n Amount: {label.split('_')[1]}" for label in labels]
    ax.set_xticklabels(labels)
    # adjust legend
    ax.legend(loc='lower center', ncols=7, title="Cancer Type", bbox_to_anchor=(0.5, -0.28))
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score_grid.png"), dpi=300)
    plt.close('all')


def create_hue_performance_plot(df: pd.DataFrame, metric: str):
    # df = df[df["cancer"] == "All"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="cancer", y=metric, hue="cancer", ax=ax)
    ax.set_title(f"{metric.upper()} score")
    ax.set_ylabel(f"{metric.upper()} score")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot(df: pd.DataFrame):
    sub_df = df[df["cancer"] == "All"].copy()

    # df melt
    sub_df = sub_df.melt(id_vars=["cancer"], value_vars=["precision", "recall", "auc"], var_name="metric",
                         value_name="score")
    # rename precision to Precision, recall to Recall and auc to AUC
    sub_df["metric"] = sub_df["metric"].replace({"precision": "Precision", "recall": "Recall", "auc": "AUC"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub_df, x="cancer", y="score", hue="metric", ax=ax)
    ax.set_title(f"Overview scores")
    ax.set_ylabel(f"Overview scores")

    ax.set_xlabel("")
    plt.legend(title="Metrics", loc='upper left')  #
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"overview_scores.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot_per_combination(df: pd.DataFrame):
    sub_df = df[df["cancer"] == "All"].copy()
    # df melt
    sub_df = sub_df.melt(id_vars=["walks"], value_vars=["precision", "recall", "auc"], var_name="metric",
                         value_name="score")
    # rename precision to Precision, recall to Recall and auc to AUC
    sub_df["metric"] = sub_df["metric"].replace({"precision": "Precision", "recall": "Recall", "auc": "AUC"})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sub_df, x="walks", y="score", hue="metric", ax=ax)
    ax.set_title(f"Overview scores")
    ax.set_ylabel(f"Overview scores")

    ax.set_xlabel("")
    plt.legend(title="Metrics", loc='upper left')  #
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"overview_scores_by_walks.png"), dpi=300)
    plt.close('all')


def create_performance_overview_heatmap(df: pd.DataFrame):
    # Filter for "All" cancer type
    sub_df = df[df["cancer"] == "All"].copy()

    # Melt the DataFrame to convert metrics into a single column
    sub_df = sub_df.melt(id_vars=["walk_distance", "amount_of_walks"],
                         value_vars=["precision", "recall", "auc", "accuracy", "f1"],
                         var_name="metric", value_name="score")

    # Rename metrics for better readability
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC", "accuracy": "Accuracy", "f1": "F1 Score"})

    # Create a heatmap for each metric
    for metric in ["Precision", "Recall", "AUC", "Accuracy", "F1 Score"]:
        metric_data = sub_df[sub_df["metric"] == metric].pivot_table(
            index="walk_distance", columns="amount_of_walks", values="score", aggfunc="mean"
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(metric_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"{metric} Scores Heatmap")
        plt.xlabel("Amount of Walks")
        plt.ylabel("Walk Distance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Path(save_folder, f"overview_scores_heatmap_{metric}.png"), dpi=300, bbox_inches="tight")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="The cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    args = parser.parse_args()

    selected_cancers = args.cancer

    logging.info(f"Using cancer types: {selected_cancers}")

    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, "performance")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load all runs from results/classifier/classification
    results = []
    # iterate over all subfolders
    cancer_folder = Path("results", "classifier", "classification", cancers)
    for run in cancer_folder.iterdir():
        if run.is_file():
            continue

        for iteration in run.iterdir():
            if iteration.is_file():
                continue

            logging.info(iteration)
            try:
                df = pd.read_csv(Path(iteration, "results.csv"))
                # load the results from the run
                results.append(df)
            except FileNotFoundError:
                continue

    # concatenate all results
    results = pd.concat(results)
    # create new column walks by combining walk distance and amount_of_walk using an underscore
    results["walks"] = results["walk_distance"].astype(str) + "_" + results["amount_of_walks"].astype(str)

    create_hue_performance_plot(results, "accuracy")
    create_hue_performance_plot(results, "f1")
    create_performance_overview_plot(results)
    create_performance_overview_plot_per_combination(results)
    create_performance_overview_heatmap(results)

    create_grid_plot(results, "accuracy")
    create_grid_plot(results, "f1")
