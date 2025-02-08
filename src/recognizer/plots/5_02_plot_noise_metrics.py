import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

color_palette = {"Text": "blue", "Image": "red", "RNA": "green", "Mutation": "purple", "BRCA": "orange",
                 "LUAD": "lime", "BLCA": "pink", "THCA": "brown", "STAD": "black", "COAD": "grey"}

save_folder = Path("figures", "recognizer")
load_folder = Path("results", "recognizer", "aggregated_metrics")

metric_mappings = {
    "A": "Accuracy",
    "P": "Precision",
    "R": "Recall",
    "F1": "f1_nonzeros"
}


def plot_bar_plot(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot individual embeddings
    sns.barplot(data=df, x="embedding", y="f1_nonzeros", hue="embedding", alpha=0.6)

    plt.savefig(Path(save_folder, "bar_plot.png"), dpi=150)


def plot_noise(df: pd.DataFrame, metric: str):
    # calculate mean for each noise
    df = df.groupby(["walk_distance", "noise"]).mean(numeric_only=True)

    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot individual embeddings
    sns.lineplot(data=df, x="walk_distance", y=metric, hue="noise", alpha=0.6)

    plt.savefig(Path(save_folder, "noise_plot.png"), dpi=150)


def noise_grid(df, metric: str, file_name: str):
    # Ensure 'noise' is treated as a categorical variable for plotting
    df["noise"] = df["noise"].astype(str)  # Convert to string to ensure proper FacetGrid behavior
    # sort the noise values
    df["noise"] = pd.Categorical(df["noise"], categories=sorted(df["noise"].unique()), ordered=True)

    # Set up Seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Create a FacetGrid with one plot per unique noise value
    g = sns.FacetGrid(df, col="noise", col_wrap=3, height=4)

    # Map the lineplot to each facet
    g.map(
        sns.lineplot,
        "walk_distance", metric, "embedding",
        palette=color_palette, alpha=0.6
    )

    # Add legend to the grid
    g.add_legend(title="Embedding")

    # Show the plots
    plt.savefig(Path(save_folder, file_name), dpi=150)


if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+')
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--multi", "-m", action="store_true", help="Use of the multi recognizer metrics")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--metric", required=True, choices=["A", "P", "R", "F1"], default="A")
    parser.add_argument("--noise", "-n", type=float, default=0.1, help="The noise to filter")

    args = parser.parse_args()
    multi: bool = args.multi
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    foundation: bool = args.foundation
    metric: str = args.metric
    selected_cancers: [str] = '_'.join(cancers)

    metric = metric_mappings[metric]

    logging.info(
        f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}, metric: {metric},"
        f" amount_of_walk_embeddings: {amount_of_walk_embeddings}")

    save_folder = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings))
    logging.info(f"Saving results to: {save_folder}")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    if multi:
        file_path = Path(load_folder, "multi", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    else:
        file_path = Path(load_folder, "simple", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    logging.info(f"Loading file using {file_path}...")
    df = pd.read_csv(file_path)
    logging.info(df)

    # remove -1 walk_distance
    df = df[df["walk_distance"] != -1]
    # filter all over 10
    df = df[df["walk_distance"] <= 10]

    # filter noise <= 0.5
    df = df[df["noise"] <= 0.5]

    # find the value 0.7606436894358242 in the df
    for index, row in df.iterrows():
        if row["f1_nonzeros"] == 0.7606436894358242:
            logging.info(row)
            input()

    # calculate mean of embeddings
    grouped = df.groupby(["walk_distance", "embedding", "noise"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    grouped = grouped.sort_values(by=["accuracy"], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    grouped = grouped.reset_index()

    logging.info(grouped)

    plot_noise(grouped, metric)

    multi_name = "multi" if multi else "simple"
    foundation_name = "foundation" if foundation else ""

    if foundation_name:
        file_name = f"{metric}_{multi_name}_{foundation_name}_noise_grid.png"
    else:
        file_name = f"{metric}_{multi_name}_noise_grid.png"

    noise_grid(grouped, metric, file_name)

    logging.info("Done")
