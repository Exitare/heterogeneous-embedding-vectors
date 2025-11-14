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


def create_line_chart(metric: Metric, grouped_df: pd.DataFrame, save_folder: Path):
    # create a line chart
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
    plt.close('all')


if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results - Zero metrics version')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--models", "-m", choices=["multi", "simple", "baseline_m", "baseline_s"], default="multi",
                        help="The model to use")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--noise_ratio", "-n", type=float, help="The noise ratio to use", default=0.0)

    args = parser.parse_args()
    model: str = args.models
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: List[str] = args.cancer
    foundation: bool = args.foundation
    selected_cancers: str = '_'.join(cancers)
    noise_ratio: float = args.noise_ratio

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

    # color palette should include only the embedding that are available in the dataset
    available_embeddings = df["embedding"].unique()
    color_palette = {k: v for k, v in color_palette.items() if k in df["embedding"].unique()}
    order = [k for k in order if k in available_embeddings]

    # Find all columns that end with "_zero" and have data
    zero_columns = [col for col in df.columns if col.endswith("_zero") and df[col].notna().any()]

    logging.info(f"Found {len(zero_columns)} zero-labeled metrics: {zero_columns}")

    # Generate plots for all zero-labeled metrics
    for zero_col in zero_columns:
        # Create a metric name for the plot (e.g., "accuracy_zero" -> "Accuracy Zero")
        metric_label = " ".join(word.capitalize() for word in zero_col.split("_"))
        zero_metric = Metric(zero_col, metric_label)

        # Group data by walk_distance and embedding
        df_grouped_by_wd_embedding = df.groupby(["walk_distance", "embedding"]).mean()
        df.reset_index(drop=True, inplace=True)

        create_line_chart(zero_metric, df_grouped_by_wd_embedding, save_folder)

    if not zero_columns:
        logging.warning("No zero-labeled metrics found in the data!")
    else:
        logging.info(f"Successfully generated {len(zero_columns)} zero-metric line plots!")