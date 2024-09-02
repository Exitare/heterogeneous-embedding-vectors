import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

save_folder = Path("figures", "recognizer")
load_folder = Path("results", "recognizer", "aggregated_metrics")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("--file_name", "-fn", type=Path, required=False, help="File name to save the plot")
    parser.add_argument("-c", "--cancer", required=False, nargs='+')
    parser.add_argument("--foundation", "-f", action="store_true", help="Plot for foundation model")
    parser.add_argument("--multi", "-m", action="store_true", help="Plot for multi recognizer")

    args = parser.parse_args()
    multi = args.multi
    file_name: Path = args.file_name
    cancers: [] = args.cancer
    foundation: bool = args.foundation

    print(f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}")

    if multi:
        selected_cancers = '_'.join(cancers)
        if foundation:
            load_folder = Path(load_folder, "mrf", selected_cancers)
            file = Path(load_folder, "split_metrics.csv")
        else:
            load_folder = Path(load_folder, "mr", selected_cancers)
            file = Path(load_folder, "metrics.csv")
    else:
        if foundation:
            load_folder = Path(load_folder, "srf")
            file = Path(load_folder, "split_metrics.csv")
        else:
            load_folder = Path(load_folder, "sr")
            file = Path(load_folder, "metrics.csv")

    print(f"Loading file {file}...")
    df = pd.read_csv(file)

    print(df)
    # calculate mean of embeddings
    df = df.groupby(["walk_distance", "embedding"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    df = df.sort_values(by=["accuracy"], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    df = df.reset_index()

    # print mean accuracy for each embedding
    print(df[["embedding", "accuracy"]].groupby("embedding").mean(numeric_only=True))
    df_mean = df.groupby("walk_distance", as_index=False)["accuracy"].mean()

    # upper case all embedding
    df["embedding"] = df["embedding"].str.upper()

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
    sns.lineplot(data=df, x="walk_distance", y="accuracy", hue="embedding", palette="tab10", alpha=0.6)

    # Plot mean line
    sns.lineplot(data=df_mean, x="walk_distance", y="accuracy", color='black', marker='o', linestyle='--', label='Mean')

    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Walk Distance")
    plt.xticks(rotation=45)
    plt.legend(title="Embedding")
    plt.tight_layout()

    if file_name is None:
        plt.show()
    else:
        plt.savefig(Path(save_folder, file_name))
