import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap

load_folder = Path("results", "classifier", "summed_embeddings")
fig_save_folder = Path("results", "classifier", "umap")

walk_distances = [3, 4, 5]
amount_of_walks = [3, 4, 5]

if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=True)
    args = parser.parse_args()

    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)

    fig_save_folder = Path(fig_save_folder, cancers)

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    load_folder = Path(load_folder, cancers)

    # Initialize the plot with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Loop over each combination of walk distance and amount of walks
    plot_idx = 0
    print("Creating figure...")
    for walk_distance in walk_distances:
        for amount_of_walk in amount_of_walks:
            subfolder = Path(load_folder, f"{walk_distance}_{amount_of_walk}")

            data = pd.read_csv(Path(subfolder, "summed_embeddings.csv"))
            # save cancer name
            available_cancers = data["cancer"]

            data.drop(columns=["patient_id", "cancer"], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.dropna(inplace=True)

            # Apply UMAP to reduce dimensions to 2D for visualization
            umap_reducer = umap.UMAP()
            df_umap = umap_reducer.fit_transform(data)

            # Convert to DataFrame for easier plotting
            df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
            df_plot['cancer'] = available_cancers

            # Plot the UMAP result in the respective subplot
            sns.scatterplot(x='UMAP1', y='UMAP2', hue='cancer',
                            palette={"BRCA": "red", "BLCA": "blue", "LUAD": "green", "STAD": "purple", "THCA": "orange",
                                     "COAD": "yellow"}, data=df_plot, s=25, ax=axes[plot_idx])
            axes[plot_idx].set_title(f'WD: {walk_distance}, AW: {amount_of_walk}')
            # remove legend
            axes[plot_idx].get_legend().remove()
            plot_idx += 1

    plt.tight_layout()
    # add legend to last plot
    axes[-1].legend(title='Cancer', loc='upper right')
    plt.suptitle('UMAP Visualization of Summed Embeddings', y=1.02)
    plt.savefig(Path(fig_save_folder, f'combined_summed_embeddings.png'), dpi=150)
    plt.close()
