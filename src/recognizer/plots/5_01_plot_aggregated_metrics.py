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
    "F1NZ": "f1_nonzeros",
    "F1Z": "f1_zeros"
}

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--multi", "-m", action="store_true", help="Use of the multi recognizer metrics")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--metric", required=True, choices=["A", "P", "R", "F1Z", "F1NZ"], default="A")

    args = parser.parse_args()
    multi: bool = args.multi
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    foundation: bool = args.foundation
    metric: str = args.metric
    selected_cancers: [str] = '_'.join(cancers)

    metric: str = metric_mappings[metric]

    logging.info(
        f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}, metric: {metric},"
        f" amount_of_walk_embeddings: {amount_of_walk_embeddings}")

    if multi:
        file_path = Path(load_folder, "multi", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    else:
        file_path = Path(load_folder, "simple", selected_cancers, str(amount_of_walk_embeddings),
                         "metrics.csv" if not foundation else "split_metrics.csv")

    logging.info(f"Loading file using {file_path}...")
    df = pd.read_csv(file_path)

    df = df[df["noise"] == 0.0]
    assert df["noise"].unique() == 0.0, "Noise is not unique"

    # drop -1 embeddings
    df = df[df["walk_distance"] != -1]
    # filter all over 10
    #df = df[df["walk_distance"] <= 10]

    # calculate mean of embeddings
    df = df.groupby(["walk_distance", "embedding"]).mean(numeric_only=True)
    print(df)
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

    # Plot individual embeddings
    sns.lineplot(data=df, x="walk_distance", y=metric, hue="embedding", palette=color_palette, alpha=0.6)

    # Plot mean line
    # sns.lineplot(data=df_mean, x="walk_distance", y=metric, color='black', marker='o', linestyle='--', label='Mean')

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("Walk Distance")
    plt.xticks(rotation=45)
    plt.legend(title="Embedding")
    plt.tight_layout()
    # set y axis to 0.2 and 1
    plt.ylim(0.2, 1)
    # start x ticks at 3
    plt.xticks([3, 4,5,6,7,8,9,10,15,20,25,30,35, 50])
    plt.xlim(3, 51)

    if multi:
        save_path = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings), "0.0")
        save_file_name: str = f"multi_{metric}" if not foundation else f"multi_foundation_{metric}.png"
    else:
        save_path = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings), "0.0")
        save_file_name: str = f"simple_{metric}" if not foundation else f"simple_foundation_{metric}.png"

    if not save_path.exists():
        save_path.mkdir(parents=True)

    plt.savefig(Path(save_path, save_file_name), dpi=150)
