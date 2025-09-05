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
fig_save_folder = Path("figures", "classifier_holdout")

color_palette = {
    "Annotation": "#c8b7b7ff",
    "Image": "#d38d5fff",
    "RNA": "#c6afe9ff",
    "Mutation": "#de87aaff",
    "BRCA": "#c837abff",
    "LUAD": "#37abc8ff",
    "BLCA": "#ffcc00ff",
    "THCA": "#d35f5fff",
    "STAD": "#f47e44d7",
    "COAD": "#502d16ff"
}

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 12
    parser = ArgumentParser(description='Embedding Clustering')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--walk_amount", "-a", required=False, type=int, help="The walk amount.", default=3)
    parser.add_argument("--walk_distance", "-w", required=False, type=int, help="The walk distance.", default=3)
    parser.add_argument("--modalities", "-m", nargs="+", default=["annotations", "images", "mutations", "rna"],
                        help="Modalities to include in the summing process.",
                        choices=["rna", "annotations", "mutations", "images"])
    args = parser.parse_args()

    cancers = args.selected_cancers
    selected_cancers = "_".join(cancers)
    walk_amount = args.walk_amount
    walk_distance = args.walk_distance
    modalities = args.modalities
    selected_modalities = "_".join(modalities)

    if len(modalities) < 2:
        raise ValueError("At least two modalities must be selected for summing embeddings.")

    fig_save_folder = Path(fig_save_folder, selected_cancers, selected_modalities, "clustering")
    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    h5_file_path = Path("results", "classifier_holdout", "summed_embeddings", selected_cancers, selected_modalities,
                        f"{walk_amount}_{walk_distance}", "0", "summed_embeddings.h5")
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

    # Manually map cancer types to colors use the color_palette
    cancer_palette = {cancer: color_palette[cancer] for cancer in unique_cancers}

    # Ensure cancer types are correctly mapped
    color_mapping = [cancer_palette[cancer] for cancer in cancer_types]

    # Plot PCA projection with true cancer labels
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mapping, alpha=0.7, edgecolors="k")

    # Manually create the legend with correct colors
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cancer_palette[c], markersize=6)
                      for c in unique_cancers]

    # plt.legend(legend_patches, unique_cancers, title="Cancer Type", loc="upper right", bbox_to_anchor=(0.9, 1))

    # plt.title(f"GMM Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"{walk_amount}_{walk_distance}_pca_clustering.png"), dpi=300)
    plt.close('all')

    # Compute cluster centers in PCA space
    centers = gmm.means_
    centers_pca = pca.transform(centers)

    # Create subplots:
    # Left: Data points colored by GMM cluster labels (model-driven segmentation)
    # Right: Data points colored by Cancer Types (data provenance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ----- Left subplot: Colored by GMM cluster labels -----
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.set_title("GMM Clustering")
    # Overlay cluster centers on this plot
    for idx, center in enumerate(centers_pca):
        ax1.scatter(center[0], center[1], marker='X', s=200, c='black')
        ax1.text(center[0], center[1], f"{idx}", fontsize=12, color='black',
                 ha='center', va='center', fontweight='bold')
    fig.colorbar(scatter1, ax=ax1, label="Cluster Label")

    # ----- Right subplot: Colored by Cancer Types (Data Provenance) -----
    for cancer in unique_cancers:
        mask = cancer_types == cancer
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=cancer_palette[cancer], alpha=0.6)
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("Cancer Type Distribution")

    # Create legend patches for the cancer types to ensure color parity
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cancer_palette[c],
                   markersize=8, label=c)
        for c in unique_cancers
    ]
    ax2.legend(handles=legend_patches, title="Cancer Type", bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0.)

    fig.suptitle(f"SC: {walk_distance} R: {walk_amount}", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"{walk_distance}_{walk_amount}_gmm_clustering.png"), dpi=300)
