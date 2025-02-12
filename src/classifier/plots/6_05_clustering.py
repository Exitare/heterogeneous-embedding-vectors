from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import logging
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fig_save_folder = Path("figures", "classifier")


if __name__ == '__main__':

    parser = ArgumentParser(description='Embedding Clustering')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--walk_amount", "-a", required=False, type=int, help="The walk amount.", default=3)
    parser.add_argument("--walk_distance", "-w", required=False, type=int, help="The walk distance.", default=3)
    args = parser.parse_args()

    cancers = args.selected_cancers
    selected_cancers = "_".join(cancers)
    walk_amount = args.walk_amount
    walk_distance = args.walk_distance

    fig_save_folder = Path(fig_save_folder, selected_cancers, "clustering")
    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    h5_file_path = Path("results", "classifier", "summed_embeddings", selected_cancers,
                        f"{walk_amount}_{walk_distance}", "summed_embeddings.h5")
    logging.info(f"Loading embeddings from {h5_file_path}")

    # Load embeddings and cancer types
    with h5py.File(h5_file_path, 'r') as h5_file:
        X_data = h5_file['X'][:]  # Load embeddings
        y_data = h5_file['y'][:]  # Load cancer types (stored as bytes)

    # Decode cancer type labels
    cancer_types = np.array([cancer.decode("utf-8") for cancer in y_data])
    unique_cancers = np.unique(cancer_types)

    logging.info(f"Extracted {X_data.shape[0]} samples with {X_data.shape[1]} features.")
    logging.info(f"Found {len(unique_cancers)} unique cancer types: {unique_cancers}")

    # Reduce dimensionality with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)

    # Fit GMM clustering
    num_clusters = len(unique_cancers)  # Use the number of unique cancer types
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(X_data)

    # Manually map cancer types to colors
    cancer_palette = dict(zip(unique_cancers, sns.color_palette("Set1", len(unique_cancers))))  # Assign colors

    # Ensure cancer types are correctly mapped
    color_mapping = [cancer_palette[cancer] for cancer in cancer_types]

    # Plot PCA projection with true cancer labels
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mapping, alpha=0.7, edgecolors="k")

    # Manually create the legend with correct colors
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cancer_palette[c], markersize=6)
                      for c in unique_cancers]

    plt.legend(legend_patches, unique_cancers, title="Cancer Type", loc="upper right", bbox_to_anchor=(0.9, 1))

    plt.title(f"GMM Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"{walk_amount}_{walk_distance}_gmm_clustering.png"), dpi=150)