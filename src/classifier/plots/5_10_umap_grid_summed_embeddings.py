import pandas as pd
import h5py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap

load_folder = Path("results", "classifier", "summed_embeddings")
fig_save_folder = Path("figures", "classifier", "umap")

if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=True)
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.", default=5)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.", default=5)
    args = parser.parse_args()

    selected_cancers = args.selected_cancers
    walk_distances = args.walk_distance
    amount_of_walks = args.amount_of_walks

    walk_distances: [int] = range(3, walk_distances + 1)
    amount_of_walks: [int] = range(3, amount_of_walks + 1)

    cancers = "_".join(selected_cancers)

    fig_save_folder = Path(fig_save_folder, cancers, f"{walk_distances[-1]}_{amount_of_walks[-1]}")

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

            h5_file_path = Path(subfolder, "summed_embeddings.h5")
            with h5py.File(h5_file_path, 'r') as h5_file:
                data = h5_file['X'][:]
                cancer_types = h5_file['y'][:].astype(str)  # Assuming 'y' is encoded as strings

            # Check if data is empty
            if data.size == 0 or cancer_types.size == 0:
                print(f"No data in {h5_file_path}. Skipping.")
                continue

            # Apply UMAP to reduce dimensions to 2D for visualization
            umap_reducer = umap.UMAP()
            df_umap = umap_reducer.fit_transform(data)

            # Convert to DataFrame for easier plotting
            df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
            df_plot['cancer'] = cancer_types

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
    axes[-1].legend(title='Cancer', loc='lower left')
    plt.suptitle('UMAP Visualization of Summed Embeddings', y=1.02)
    plt.savefig(Path(fig_save_folder, f'combined_summed_embeddings.png'), dpi=150)
    plt.close()
