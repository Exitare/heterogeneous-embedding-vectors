import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import logging
from helper.load_metric_data import load_metric_data
from helper.plot_styling import color_palette
from typing import List

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


def noise_grid(df, metric: str, file_name: str):
    # Ensure 'noise' is treated as a categorical variable for plotting
    # convert noise to percentage
    tmp_df = df.copy()

    # change text to Annotation
    tmp_df["embedding"] = tmp_df["embedding"].replace("Text", "Annotation")

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
    g.set_xlabels("Sample Count")
    g.set_ylabels(metric.upper())
    g.set(ylim=(-0.1, 1.02))
    # set title for each plot colname %
    g.set_titles(col_template="{col_name} %")
    # Add legend to the grid
    g.add_legend(title="Modality")

    # Show the plots
    plt.savefig(Path(save_folder, file_name), dpi=300)


def reduced_noise_grid(df, metric: str, file_name: str, title: str):
    # Ensure 'noise' is treated as a categorical variable for plotting

    tmp_df = df.copy()

    # change text to Annotation
    tmp_df["embedding"] = tmp_df["embedding"].replace("Text", "Annotation")
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
    g.set_xlabels("Sample Count")
    g.set_ylabels(metric.upper())
    # set y-lim from 0 to 1
    g.set(ylim=(-0.1, 1))

    # set title for each plot colname %
    g.set_titles(col_template="{col_name} %")
    # Add legend to the grid
    g.add_legend(title="Modality")
    # set title
    g.figure.suptitle(title, y=0.95, fontsize=12)
    # move subtitlte to the left
    g.figure.subplots_adjust(left=0.14)

    # Adjust layout to prevent subtitle cutoff
    plt.subplots_adjust(top=0.85)

    # Show the plots
    plt.savefig(Path(save_folder, file_name), dpi=300, bbox_inches="tight")


def plot_embedding_heatmap(df: pd.DataFrame, metric: str, file_name: str, noise_ratio: float):
    """Create a heatmap showing mean metric for each embedding (modality and cancer type) across sample counts."""
    # Define modalities and cancer types
    modalities = ["Annotation", "Image", "RNA", "Mutation"]
    cancer_types = ["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
    all_embeddings = modalities + cancer_types

    # Replace Text with Annotation
    tmp_df = df.copy()
    tmp_df["embedding"] = tmp_df["embedding"].replace("Text", "Annotation")
    
    # Filter by the specified noise ratio
    if "noise" in tmp_df.columns:
        tmp_df = tmp_df[tmp_df["noise"] == noise_ratio].copy()
        if tmp_df.empty:
            logging.warning(f"No data found for noise ratio {noise_ratio}")
            return

    # Only keep relevant embeddings
    tmp_df = tmp_df[tmp_df["embedding"].isin(all_embeddings)].copy()
    if tmp_df.empty:
        logging.info("No embeddings found for heatmap plot.")
        return

    # Group by embedding and walk_distance, calculate mean
    grouped = tmp_df.groupby(["embedding", "walk_distance"])[metric].mean().reset_index()

    # Pivot for heatmap: embeddings as rows, sample counts as columns
    heatmap_data = grouped.pivot(index="embedding", columns="walk_distance", values=metric)
    # Reorder rows: modalities first, then cancer types
    ordered_rows = [e for e in all_embeddings if e in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(ordered_rows)

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={
            'label': metric.upper(),
            'shrink': 0.8,  # Reduce colorbar size to decrease gap
            'pad': 0.02     # Reduce padding between heatmap and colorbar
        },
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 14}
    )
    
    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # Increase tick label size
    cbar.set_label(metric.upper(), fontsize=14)  # Increase colorbar label size
    
    #plt.title(f"{metric.upper()} Heatmap: Modalities + Cancer Types Across Sample Counts (Noise: {int(noise_ratio*100)}%)", fontsize=16, fontweight='bold')
    plt.xlabel("Sample Count", fontsize=14)
    plt.ylabel("Embedding", fontsize=14)
    plt.tight_layout()
    
    # Set tick parameters with rotation AFTER tight_layout
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=12)
    
    plt.savefig(Path(save_folder, file_name), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {file_name}")


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
    cancers: List[str] = args.cancer
    foundation: bool = args.foundation
    metric: str = args.metric
    noise_ratio: float = args.noise
    selected_cancers: str = '_'.join(cancers)

    metric_input = metric
    metric = metric_mappings[metric]

    logging.info(
        f"Loading data for multi: {multi}, cancers: {cancers}, foundation: {foundation}, metric: {metric},"
        f" amount_of_walk_embeddings: {amount_of_walk_embeddings}, noise_ratio: {noise_ratio}")

    multi_name = "multi" if multi else "simple"
    foundation_name = "foundation" if foundation else ""
    
    # Build save folder path based on model type and foundation status
    if foundation_name:
        model_folder = f"{multi_name}_{foundation_name}"
    else:
        model_folder = multi_name
    
    save_folder = Path(save_folder, selected_cancers, str(amount_of_walk_embeddings), str(noise_ratio), model_folder)
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
    # logging.info(df)
    # print walk_distance == 3 and noise == 0.1 and embedding text
    print(df[(df["walk_distance"] == 3) & (df["noise"] == 0.5) & (df["embedding"] == "Text")])

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

    if foundation_name:
        file_name = f"{metric}_{multi_name}_{foundation_name}_noise_grid.png"
        reduced_file_name = f"{metric}_{multi_name}_{foundation_name}_reduced_noise_grid.png"
        heatmap_file = f"{metric}_{multi_name}_{foundation_name}_embedding_heatmap.png"
        figure_title = f"{multi_name.replace('_', ' ').capitalize()} {foundation_name.capitalize()}"
    else:
        file_name = f"{metric}_{multi_name}_noise_grid.png"
        reduced_file_name = f"{metric}_{multi_name}_reduced_noise_grid.png"
        heatmap_file = f"{metric}_{multi_name}_embedding_heatmap.png"
        figure_title = f"{multi_name.replace('_', ' ').capitalize()}"

    logging.info(f"Saving to {save_folder / file_name}")
    logging.info(f"Saving to {save_folder / reduced_file_name}")
    logging.info(f"Saving to {save_folder / heatmap_file}")
    noise_grid(df, metric, file_name)
    reduced_noise_grid(df, metric, reduced_file_name, figure_title)

    # Add embedding heatmap (modalities + cancer types) - filtered by noise ratio
    plot_embedding_heatmap(df, metric, heatmap_file, noise_ratio)

    logging.info("Done")
