import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="Data containing the recognizer results")

    args = parser.parse_args()
    data_folder: Path = args.data_folder

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
    plt.show()
