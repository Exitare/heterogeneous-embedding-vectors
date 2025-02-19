import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import logging
from helper.load_metric_data import load_metric_data
from helper.plot_styling import color_palette

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


save_folder = Path("figures", "recognizer")
load_folder = Path("results", "recognizer")

metric_mappings = {
    "A": "Accuracy",
    "P": "Precision",
    "R": "Recall",
    "F1": "f1",
    "F1Z": "f1_zeros",
    "BA": "balanced_accuracy",
    "MCC": "mcc"
}


def plot_bar_plot(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot individual embeddings
    sns.barplot(data=df, x="embedding", y="f1", hue="embedding", alpha=0.6)

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
    # convert noise to percentage
    tmp_df = df.copy()

    tmp_df["noise"] = tmp_df["noise"] * 100
    # to int
    tmp_df["noise"] = tmp_df["noise"].astype(int)
    tmp_df["noise"] = tmp_df["noise"].astype(str)  # Convert to string to ensure proper FacetGrid behavior
    # sort the noise values
    tmp_df["noise"] = pd.Categorical(tmp_df["noise"], categories=sorted(tmp_df["noise"].unique()), ordered=True)

    # Set up Seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Create a FacetGrid with one plot per unique noise value
    g = sns.FacetGrid(tmp_df, col="noise", col_wrap=3, height=4)

    # Map the lineplot to each facet, increase line
    g.map(
        sns.lineplot,
        "walk_distance", metric, "embedding", markers=True,
        palette=color_palette, alpha=0.6, marker="o", markersize=5, linewidth=1.5
    )
    # change y axis to metric
    g.set_ylabels(metric_input)
    # change x axis to walk_distance
    g.set_xlabels("Walk Distance")
    g.set_ylabels(metric.upper())
    g.set(ylim=(0, 1.02))
    # set title for each plot colname %
    g.set_titles(col_template="{col_name} %")
    # Add legend to the grid
    g.add_legend(title="Modality")

    # Show the plots
    plt.savefig(Path(save_folder, file_name), dpi=150)


def reduced_noise_grid(df, metric: str, file_name: str):
    # Ensure 'noise' is treated as a categorical variable for plotting

    tmp_df = df.copy()
    # convert noise to percentage
    tmp_df["noise"] = tmp_df["noise"] * 100
    # to int
    tmp_df["noise"] = tmp_df["noise"].astype(int)
    tmp_df["noise"] = tmp_df["noise"].astype(str)  # Convert to string to ensure proper FacetGrid behavior

    # only select noise values of 10 and 50
    tmp_df = tmp_df[(tmp_df["noise"] == "10") | (tmp_df["noise"] == "50")]
    # sort the noise values
    tmp_df["noise"] = pd.Categorical(tmp_df["noise"], categories=sorted(tmp_df["noise"].unique()), ordered=True)

    # Set up Seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Create a FacetGrid with one plot per unique noise value
    g = sns.FacetGrid(tmp_df, col="noise", col_wrap=2, height=4)

    # Map the lineplot to each facet, increase line
    g.map(
        sns.lineplot,
        "walk_distance", metric, "embedding", markers=True,
        palette=color_palette, alpha=0.6, marker="o", markersize=5, linewidth=1.5
    )
    # change y axis to metric
    g.set_ylabels(metric_input)
    # change x axis to walk_distance
    g.set_xlabels("Walk Distance")
    g.set_ylabels(metric.upper())
    #set y-lim from 0 to 1
    g.set(ylim=(0, 1))

    # set title for each plot colname %
    g.set_titles(col_template="{col_name} %")
    # Add legend to the grid
    g.add_legend(title="Modality")

    # Show the plots
    plt.savefig(Path(save_folder, file_name), dpi=150)


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--multi", "-m", action="store_true", help="Use of the multi recognizer metrics")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")
    parser.add_argument("--metric", required=True, choices=["A", "P", "R", "F1", "F1Z", "BA", "MCC"], default="A")
    parser.add_argument("--noise", "-n", type=float, default=0.1, help="The noise to filter")

    args = parser.parse_args()
    multi: bool = args.multi
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    foundation: bool = args.foundation
    metric: str = args.metric
    selected_cancers: [str] = '_'.join(cancers)

    metric_input = metric
    metric = metric_mappings[metric]

    logging.info(
        f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}, metric: {metric},"
        f" amount_of_walk_embeddings: {amount_of_walk_embeddings}")

    save_folder = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings))
    logging.info(f"Saving results to: {save_folder}")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    if multi:
        load_path = Path(load_folder, "multi", selected_cancers, str(amount_of_walk_embeddings))

    else:
        load_path = Path(load_folder, "simple", selected_cancers, str(amount_of_walk_embeddings))

    if not save_folder.exists():
        save_folder.mkdir(parents=True)


    logging.info(f"Loading files using {load_path}...")

    df = load_metric_data(load_folder=load_path, noise_ratio=-1, foundation=foundation)
    #logging.info(df)
    # print walk_distance == 3 and noise == 0.1 and embedding text
    #print(df[(df["walk_distance"] == 3) & (df["noise"] == 0.1) & (df["embedding"] == "Text")])

    # remove -1 walk_distance
    df = df[df["walk_distance"] != -1]
    # filter all over 10
    df = df[df["walk_distance"] <= 10]

    # filter noise <= 0.5
    df = df[df["noise"] <= 0.5]

    df.reset_index(drop=True, inplace=True)

    # calculate mean of embeddings
    grouped = df.groupby(["walk_distance", "embedding", "noise"]).mean(numeric_only=True)
    # embeddings,iteration,embedding,accuracy,precision,recall,f1
    # plot the accuracy for each embeddings, hue by embeddings
    grouped = grouped.sort_values(by=["accuracy"], ascending=False)

    # plot line plot for embeddings, embeddings and accuracy
    grouped = grouped.reset_index()

    plot_bar_plot(grouped)


    plot_noise(grouped, metric)

    multi_name = "multi" if multi else "simple"
    foundation_name = "foundation" if foundation else ""

    if foundation_name:
        file_name = f"{metric}_{multi_name}_{foundation_name}_noise_grid.png"
        reduced_file_name = f"{metric}_{multi_name}_{foundation_name}_reduced_noise_grid.png"
    else:
        file_name = f"{metric}_{multi_name}_noise_grid.png"
        reduced_file_name = f"{metric}_{multi_name}_reduced_noise_grid.png"

    logging.info(f"Saving to {file_name}")
    logging.info(f"Saving to {reduced_file_name}")
    noise_grid(df, metric, file_name)
    reduced_noise_grid(df, metric, reduced_file_name)

    logging.info("Done")
