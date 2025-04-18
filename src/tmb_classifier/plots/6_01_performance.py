from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging

save_folder = Path("figures", "tmb_classifier")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "Matthews Correlation Coefficient",
    "balanced_accuracy": "Balanced Accuracy",
}

color_palette = {
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
    "Low": "#c837abff",
    "High": "#37abc8ff",
    "N/A": "#ffcc00ff",
    "3_3": "#c837abff",
    "3_4": "#ffcc00ff",
    "3_5": "#37abc8ff",
    "4_3": "#e63946ff",
    "4_4": "#2a9d8fff",
    "4_5": "#f4a261ff",
    "5_3": "#264653ff",
    "5_4": "#8ecae6ff",
    "5_5": "#ff6700ff",
    "6_6": "#6a0572ff"
}

def create_grid_plot(df: pd.DataFrame, metric: str):
    # create a grid plot for the accuracy, each unique value of walks should have a separate plot, the hue is cancer
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=df, x="tmb", y=metric, hue="walks", ax=ax, palette=color_palette,
                  order=["Low", "High", "N/A", "All"], hue_order=["3_3", "3_4", "3_5", "4_3", "4_4", "4_5", "5_3", "5_4", "5_5"])
    ax.set_title(f"{metrics[metric]} Score")
    ax.set_ylabel(f"{metrics[metric]} Score")
    ax.set_xlabel("")

    # set ylim
    ax.set_ylim(0, 1.01)

    # change legend by removing the underscore
    handles, labels = ax.get_legend_handles_labels()
    # convert to Sample count: 3 Repeats: 3
    labels = [f"SC: {label.split('_')[0]} R: {label.split('_')[1]}" for label in labels]
    ax.legend(handles, labels, title="Tumor Mutational Burden", ncols=5, loc='lower center', bbox_to_anchor=(0.5, -0.3))

    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #labels = [f"Sample Count: {label.split('_')[0]}\n Amount: {label.split('_')[1]}" for label in labels]
    #ax.set_xticklabels(labels)
    # adjust legend
    #ax.legend(loc='lower center', ncols=7, title="Tumor Mutation Burden", bbox_to_anchor=(0.5, -0.28))
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score_grid.png"), dpi=300)
    plt.close('all')


def create_hue_performance_plot(df: pd.DataFrame, metric: str):
    # df = df[df["cancer"] == "All"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="tmb", y=metric, hue="tmb", ax=ax, palette=color_palette)
    ax.set_title(f"{metric.upper()} score")
    ax.set_ylabel(f"{metric.upper()} score")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot(df: pd.DataFrame):
    sub_df = df[df["tmb"] == "All"].copy()

    # df melt
    sub_df = sub_df.melt(id_vars=["tmb"], value_vars=["precision", "recall", "auc"], var_name="metric",
                         value_name="score")
    # rename precision to Precision, recall to Recall and auc to AUC
    sub_df["metric"] = sub_df["metric"].replace({"precision": "Precision", "recall": "Recall", "auc": "AUC"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub_df, x="tmb", y="score", hue="metric", ax=ax)
    ax.set_title(f"Overview scores")
    ax.set_ylabel(f"Overview scores")

    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"overview_scores.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot_per_combination(df: pd.DataFrame):
    sub_df = df[df["tmb"] == "All"].copy()
    # df melt
    sub_df = sub_df.melt(id_vars=["walks"], value_vars=["precision", "recall", "auc", "f1", "mcc", "balanced_accuracy"],
                         var_name="metric",
                         value_name="score")
    # rename precision to Precision, recall to Recall and auc to AUC
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC", "accuracy": "Accuracy", "f1": "F1 Score",
         "mcc": "MCC", "balanced_accuracy": "Balanced Accuracy"})


    # replace _ in x axis with space
    sub_df["walks"] = sub_df["walks"].str.replace("_", " ")

    # change

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sub_df, x="walks", y="score", hue="metric", ax=ax)
    ax.set_title(f"Metrics")
    ax.set_ylabel(f"")

    ax.set_xlabel("")
    ax.set_ylim(0, 1)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [f"Sample Count: {label.split(' ')[0]}\n Repeats: {label.split(' ')[1]}" for label in labels]
    ax.set_xticklabels(labels)

    plt.legend(title="", loc='lower right', ncols=7)  #
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"overview_scores_by_walks.png"), dpi=300)
    plt.close('all')


def create_performance_overview_heatmap(df: pd.DataFrame):
    # Filter for "All" cancer type
    sub_df = df[df["tmb"] == "All"].copy()

    # Melt the DataFrame to convert metrics into a single column
    sub_df = sub_df.melt(id_vars=["walk_distance", "amount_of_walks"],
                         value_vars=["precision", "recall", "auc", "accuracy", "f1", "mcc", "balanced_accuracy"],
                         var_name="metric", value_name="score")

    # Rename metrics for better readability
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC", "accuracy": "Accuracy", "f1": "F1 Score",
         "mcc": "MCC", "balanced_accuracy": "Balanced Accuracy"})

    # Create a heatmap for each metric
    for metric in ["Precision", "Recall", "AUC", "Accuracy", "F1 Score", "MCC", "Balanced Accuracy"]:
        metric_data = sub_df[sub_df["metric"] == metric].pivot_table(
            index="walk_distance", columns="amount_of_walks", values="score", aggfunc="mean"
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(metric_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5,vmin=0.2, vmax=1.0)
        # adjust heatmap range to 0.8 - 1.0

        plt.title(f"{metric} Scores Heatmap")
        plt.xlabel("Sample Count")
        plt.ylabel("Repeats")
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
    cancer_folder = Path("results", "tmb_classifier", "classification", cancers)
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

    # rename tmb values 0 to Low, 1 to High, 2 to None
    results["tmb"] = results["tmb"].replace({"0": "Low", "1": "High", "2": "N/A"})

    # create new column walks by combining walk distance and amount_of_walk using an underscore
    results["walks"] = results["walk_distance"].astype(str) + "_" + results["amount_of_walks"].astype(str)
    results.reset_index(drop=True, inplace=True)

    create_hue_performance_plot(results, "accuracy")
    create_hue_performance_plot(results, "f1")
    create_performance_overview_plot(results)
    create_performance_overview_plot_per_combination(results)
    create_performance_overview_heatmap(results)

    create_grid_plot(results, "accuracy")
    create_grid_plot(results, "f1")
