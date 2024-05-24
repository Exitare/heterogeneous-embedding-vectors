import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

save_folder = Path("figures", "performance")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="Data containing the recognizer results")
    parser.add_argument("--file_name", "-fn", type=Path, required=False, help="File name to save the plot")

    args = parser.parse_args()
    data_folder: Path = args.data
    file_name: Path = args.file_name

    df = pd.read_csv(data_folder)

    # calculate mean of embeddings
    df = df.groupby(["embeddings", "embedding"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    df = df.sort_values(by=["accuracy"], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    df = df.reset_index()

    # plot
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.lineplot(data=df, x="embeddings", y="accuracy", hue="embedding")
    plt.title("Mean accuracy for each embedding")
    plt.ylabel("Accuracy")
    plt.xlabel("Embedding")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(Path(save_folder, file_name))
