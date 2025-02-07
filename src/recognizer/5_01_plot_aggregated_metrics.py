import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import logging

from seaborn import color_palette

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

save_folder = Path("figures", "recognizer")
load_folder = Path("results", "recognizer", "aggregated_metrics")

metric_mappings = {
    "A": "Accuracy",
    "P": "Precision",
    "R": "Recall",
    "F1": "f1_nonzeros"
}

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+')
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--multi", "-m", action="store_true", help="Use of the multi recognizer metrics")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--metric", required=True, choices=["A", "P", "R", "F1"], default="A")
    parser.add_argument("--noise", "-n", type=float, default=0.0, help="The noise to filter")

    args = parser.parse_args()
    multi: bool = args.multi
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    foundation: bool = args.foundation
    metric: str = args.metric
    noise: float = args.noise
    selected_cancers: [str] = '_'.join(cancers)

    metric: str = metric_mappings[metric]

    logging.info(
        f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}, metric: {metric},"
        f" amount_of_walk_embeddings: {amount_of_walk_embeddings}, noise: {noise}")

    if multi:
        file_path = Path(load_folder, "multi", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    else:
        file_path = Path(load_folder, "simple", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    logging.info(f"Loading file using {file_path}...")
    df = pd.read_csv(file_path)

    df = df[df["noise"] == noise]
    logging.info(df)
    assert df["noise"].unique() == noise, "Noise is not unique"

    # calculate mean of embeddings
    df = df.groupby(["walk_distance", "embedding"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    df = df.sort_values(by=[metric], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    df = df.reset_index()

    # print mean accuracy for each embedding
    logging.info(df[["embedding", metric]].groupby("embedding").mean(numeric_only=True))
    df_mean = df.groupby("walk_distance", as_index=False)[metric].mean()

    title = ''

    if cancers is not None:
        title = f"Mean accuracy of walk distances using cancer\n{' '.join([can for can in cancers])}"
    else:
        title = "Mean accuracy of walk distances"

    # Plot
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    color_palette = {"Text": "blue", "Image": "red", "RNA": "green", "Mutation": "purple", "BRCA": "orange",
                     "LUAD": "lime", "BLCA": "pink", "THCA": "brown", "STAD": "black", "COAD": "grey"}

    # Plot individual embeddings
    sns.lineplot(data=df, x="walk_distance", y=metric, hue="embedding", palette=color_palette, alpha=0.6)

    # Plot mean line
    #sns.lineplot(data=df_mean, x="walk_distance", y=metric, color='black', marker='o', linestyle='--', label='Mean')

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("Walk Distance")
    plt.xticks(rotation=45)
    plt.legend(title="Embedding")
    plt.tight_layout()
    # set y axis to 0.2 and 1
    plt.ylim(0.2, 1)
    # if multi set x ticks starting at 2 to 20
    if multi:
        plt.xticks(range(3, 11))
        plt.xlim(3, 10)
    else:
        plt.xticks(range(3, 11))
        plt.xlim(3, 10)

    if multi:
        save_path = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings), str(noise))
        save_file_name: str = "multi" if not foundation else "multi_foundation.png"
    else:
        save_path = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings), str(noise))
        save_file_name: str = "simple" if not foundation else "simple_foundation.png"

    if not save_path.exists():
        save_path.mkdir(parents=True)

    plt.savefig(Path(save_path, save_file_name), dpi=150)
