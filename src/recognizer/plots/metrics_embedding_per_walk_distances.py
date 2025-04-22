from argparse import ArgumentParser
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import namedtuple
from helper.load_metric_data import load_metric_data
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_folder = Path("results", "recognizer")

color_palette = {"Text": "blue", "Image": "red", "RNA": "green", "Mutation": "purple", "BRCA": "orange",
                 "LUAD": "lime", "BLCA": "pink", "THCA": "brown", "STAD": "black", "COAD": "grey"}
order = ["Text", "Image", "RNA", "Mutation", "BRCA", "LUAD", "BLCA", "THCA", "STAD", "COAD"]

Metric = namedtuple("Metric", ["name", "label"])

metrics = {
    "A": Metric("accuracy", "Accuracy"),
    "P": Metric("precision", "Precision"),
    "R": Metric("recall", "Recall"),
    "F1": Metric("f1", "F1"),
    "F1Z": Metric("f1_zeros", "F1 Zero"),
    "BA": Metric("balanced_accuracy", "Balanced Accuracy"),
    "MCC": Metric("mcc", "Matthews Correlation Coefficient")
}


def create_bar_chart(metric: Metric, grouped_df: pd.DataFrame, df: pd.DataFrame, save_folder: Path):
    plt.figure(figsize=(10, 6))

    # Bar plot
    ax = sns.barplot(x="walk_distance", y=metric.name, hue="embedding", data=grouped_df,
                     palette=color_palette, hue_order=order, alpha=0.8, edgecolor="black")

    # Scatter plot overlay (showing all data points)
    sns.stripplot(x="walk_distance", y=metric.name, hue="embedding", data=df,
                  palette=color_palette, hue_order=order, jitter=True, dodge=True, alpha=0.5, marker="o", size=6)

    # Set title and labels
    ax.set_title(f"{metric.label} per Walk Distance and Modality")
    ax.set_ylabel(metric.label)
    ax.set_xlabel("Walk Distance")

    # set y lim between 0 and 1
    ax.set_ylim(-0.1, 1.1)

    # Remove duplicate legends from scatter plot
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(order)], labels[:len(order)], title="Embedding", loc='upper left', bbox_to_anchor=(1, 1))

    # Improve layout and show the plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_bar_chart.png"), dpi=300)


def create_line_chart(metric: Metric, grouped_df: pd.DataFrame, save_folder: Path):
    # create a line chart too
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x="walk_distance", y=metric.name, hue="embedding", data=grouped_df,
                      palette=color_palette, hue_order=order)

    # Set title and labels
    ax.set_title(f"{metric.label} per Sample Counts and Modality")
    ax.set_ylabel(metric.label)
    ax.set_xlabel("Sample Count")
    # put legend outside of plot
    plt.legend(title="Embedding", loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.1, 1.05)
    # Improve layout and show the plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_line_plot.png"), dpi=300)

def create_box_plot(metric: Metric, df: pd.DataFrame, save_folder: Path):
    # create a line chart too
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="walk_distance", y=metric.name, hue="embedding", data=df,
                      palette=color_palette, hue_order=order)

    # Set title and labels
    ax.set_title(f"{metric.label} per Sample Count and Modality")
    ax.set_ylabel(metric.label)
    ax.set_xlabel("Sample Count")
    # put legend outside of plot
    plt.legend(title="Embedding", loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0.8, 1.01)

    # Improve layout and show the plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_box_plot.png"), dpi=300)


if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--model", "-m", choices=["multi", "simple", "baseline_m", "baseline_s"], default="multi",
                        help="The model to use")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--noise_ratio", "-n", type=float, help="The noise ratio to use", default=0.0)
    parser.add_argument("--selected_metric", "-sm", required=True, choices=["A", "P", "R", "F1", "F1Z", "BA", "MCC"],
                        default="A")

    args = parser.parse_args()
    model: str = args.model
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: List[str] = args.cancer
    foundation: bool = args.foundation
    selected_cancers: str = '_'.join(cancers)
    noise_ratio: float = args.noise_ratio
    selected_metric: str = args.selected_metric

    metric = metrics[selected_metric]

    logging.info(
        f"Loading data for model: {model}, cancers: {cancers}, foundation: {foundation}, amount_of_walk_embeddings: {amount_of_walk_embeddings},"
        f" noise_ratio: {noise_ratio}")

    if model == "baseline_m":
        model = "baseline/multi"
    elif model == "baseline_s":
        model = "baseline/simple"

    load_folder = Path(load_folder, model, selected_cancers, str(amount_of_walk_embeddings))

    # load data
    df = load_metric_data(load_folder=load_folder, noise_ratio=noise_ratio, foundation=foundation)

    if "baseline" in model:
        # rename modality to embedding
        df.rename(columns={"modality": "embedding"}, inplace=True)

    print(df)

    # only select up to 10 Sample Count
    df = df[df["walk_distance"] <= 10]

    if "noise" in df.columns:
        # assert only the selected noise ratio is in the df
        assert df["noise"].unique() == noise_ratio, "Noise is not unique"

    save_folder = Path("figures", "recognizer", selected_cancers, str(amount_of_walk_embeddings), str(noise_ratio),
                       model if not foundation else f"{model}_foundation")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # color palette should inlcude only the mebedding that are availabe in the dataset

    available_embeddings = df["embedding"].unique()
    color_palette = {k: v for k, v in color_palette.items() if k in df["embedding"].unique()}
    order = [k for k in order if k in available_embeddings]

    # create bar plot for mcc for each walk_distance and modality
    df_grouped_by_wd_embedding = df.groupby(["walk_distance", "embedding"]).mean()
    df.reset_index(drop=True, inplace=True)

    create_bar_chart(metric, df_grouped_by_wd_embedding, df, save_folder)
    create_line_chart(metric, df_grouped_by_wd_embedding, save_folder)
    create_box_plot(metric, df, save_folder)
