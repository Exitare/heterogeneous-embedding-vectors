import sys

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

save_folder = Path("figures", "recognizer")
load_folder = Path("results", "recognizer", "aggregated_metrics")

if not save_folder.exists():
    save_folder.mkdir(parents=True)


def plot_bar_plot(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot individual embeddings
    sns.barplot(data=df, x="embedding", y="accuracy", hue="embedding", alpha=0.6)

    plt.savefig(Path(save_folder, "bar_plot.png"), dpi=150)


def plot_noise(df: pd.DataFrame):
    # calculate mean for each noise
    df = df.groupby(["walk_distance", "noise"]).mean(numeric_only=True)

    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot individual embeddings
    sns.lineplot(data=df, x="walk_distance", y="accuracy", hue="noise", palette="tab10", alpha=0.6)

    plt.savefig(Path(save_folder, "noise_plot.png"), dpi=150)


def noise_grid(df):
    # Ensure 'noise' is treated as a categorical variable for plotting
    df["noise"] = df["noise"].astype(str)  # Convert to string to ensure proper FacetGrid behavior
    # sort the noise values
    df["noise"] = pd.Categorical(df["noise"], categories=sorted(df["noise"].unique()), ordered=True)

    # Set up Seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Create a FacetGrid with one plot per unique noise value
    g = sns.FacetGrid(df, col="noise", col_wrap=4, height=4)

    # Map the lineplot to each facet
    g.map(
        sns.lineplot,
        "walk_distance", "accuracy", "embedding",
        palette="tab10", alpha=0.6
    )

    # Add legend to the grid
    g.add_legend(title="Embedding")

    # Show the plots
    plt.savefig(Path(save_folder, "noise_grid.png"), dpi=150)


if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=True, nargs='+')
    parser.add_argument("--multi", "-m", action="store_true", help="Plot for multi recognizer")
    parser.add_argument("-a", "--amount_of_summed_embeddings", type=int, default=1000,
                        help="Amount of summed embeddings")
    parser.add_argument("-combined", "--combined", action="store_true", help="Plot for combined model")

    args = parser.parse_args()
    multi = args.multi
    cancers: [] = args.cancer
    selected_cancers = '_'.join(cancers)
    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    combined: bool = args.combined

    print(f"Loading data for multi: {multi}, cancers: {cancers}")

    if not multi:
        load_folder = Path(load_folder, selected_cancers, "simple",
                           str(amount_of_summed_embeddings), "metrics.csv")
        save_folder = Path(save_folder, selected_cancers, "simple",
                           str(amount_of_summed_embeddings), "combined" if combined else "")
    else:
        load_folder = Path(load_folder, selected_cancers, "multi",
                           str(amount_of_summed_embeddings), "metrics.csv")
        save_folder = Path(save_folder, selected_cancers, "multi",
                           str(amount_of_summed_embeddings), "combined" if combined else "")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    df = pd.read_csv(load_folder)

    if combined:
        df = df[df["walk_distance"] == -1]
        plot_bar_plot(df)
        sys.exit(0)
    else:
        # remove all embeddings smaller than 3
        df = df[df["walk_distance"] >= 3]
        # remove noise 0.7,0.8,0.9,0.99
        df = df[~df["noise"].isin([0.7, 0.8, 0.9, 0.99])]
        # assert that no walk_distance of -1 is in the dataframe
        assert -1 not in df["walk_distance"].values, "Walk distance of -1 is present"

    # calculate mean of embeddings
    grouped = df.groupby(["walk_distance", "embedding", "noise"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    grouped = grouped.sort_values(by=["accuracy"], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    grouped = grouped.reset_index()

    plot_noise(grouped)
    noise_grid(grouped)

    print("Done")
