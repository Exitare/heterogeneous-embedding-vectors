from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FIG_ROOT = Path("figures", "single_modality_classifier")

COLOR_PALETTE = {
    "Annotation": "#c8b7b7ff",
    "Image": "#d38d5fff",
    "RNA": "#c6afe9ff",
    "Mutation": "#de87aaff",
    "BRCA": "#c837abff",
    "LUAD": "#37abc8ff",
    "BLCA": "#ffcc00ff",
    "THCA": "#d35f5fff",
    "STAD": "#f47e44d7",
    "COAD": "#502d16ff",
}

def decode_bytes_array(arr):
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])

def main():
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 12

    parser = ArgumentParser(description='Embedding Clustering (PCA + GMM) on a selected modality')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--selected_modality", "-sm", type=str,
                        choices=["rna", "annotations", "mutations", "images"], required=True,
                        help="Which modality file to cluster (matches the embedding generation output).")
    parser.add_argument("--load_root", "-l", type=str, default="results/single_modality_classifier/summed_embeddings",
                        help="Root folder where {CANCERS}/{modality}_embeddings.h5 lives.")
    parser.add_argument("--n_components", "-k", type=int, default=None,
                        help="Override number of GMM clusters (default: number of unique cancers).")
    args = parser.parse_args()

    cancers = args.selected_cancers
    selected_cancers = "_".join(cancers)
    modality = args.selected_modality
    load_root = Path(args.load_root)

    # Input path produced by your single-modality embedding writer
    h5_file_path = Path(load_root, selected_cancers, f"{modality}_embeddings.h5")
    logging.info(f"Loading embeddings from {h5_file_path}")

    if not h5_file_path.exists():
        raise FileNotFoundError(f"Could not find: {h5_file_path}")

    # Output figure folder
    fig_save_folder = Path(FIG_ROOT, selected_cancers, modality, "clustering")
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    with h5py.File(h5_file_path, 'r') as h5_file:
        X = h5_file['X'][:]
        raw_y = h5_file['y'][:]
        feature_dim = int(h5_file.attrs['feature_shape'])
        file_classes = h5_file.attrs['classes'][:]
        file_classes = decode_bytes_array(file_classes)

    y = decode_bytes_array(raw_y)
    uniq_cancers = np.unique(y)

    logging.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features (feature_shape attr={feature_dim}).")
    logging.info(f"Found {len(uniq_cancers)} unique cancer types: {uniq_cancers.tolist()}")

    # Sanity: feature dim matches attr
    if X.shape[1] != feature_dim:
        logging.warning(f"Feature dimension {X.shape[1]} != feature_shape attr {feature_dim}")

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # GMM clustering
    num_clusters = args.n_components if args.n_components is not None else len(uniq_cancers)
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(X)

    # Color mapping for cancer labels
    cancer_palette = {c: COLOR_PALETTE.get(c, None) for c in uniq_cancers}
    color_mapping = [cancer_palette.get(c, None) for c in y]

    # ---- Figure 1: PCA colored by true cancer labels ----
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_mapping, alpha=0.7, edgecolors="k")
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cancer_palette[c], markersize=6, label=c)
        for c in uniq_cancers
    ]
    plt.legend(handles=legend_patches, title="Cancer Type", loc="best", frameon=True)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA by Cancer Type • modality={modality}")
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"{modality}_pca_by_cancer.png"), dpi=300)
    plt.close('all')

    # ---- Figure 2: Side-by-side (GMM clusters vs cancer labels) ----
    centers_pca = pca.transform(gmm.means_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: by GMM cluster label
    s1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.set_title(f"GMM Clustering (k={num_clusters})")
    for idx, center in enumerate(centers_pca):
        ax1.scatter(center[0], center[1], marker='X', s=200, c='black')
        ax1.text(center[0], center[1], f"{idx}", fontsize=12, color='black',
                 ha='center', va='center', fontweight='bold')
    fig.colorbar(s1, ax=ax1, label="Cluster")

    # Right: by cancer labels
    for c in uniq_cancers:
        mask = (y == c)
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], color=cancer_palette[c], alpha=0.6, label=c)
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("Cancer Type Distribution")
    ax2.legend(title="Cancer Type", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.suptitle(f"Modality: {modality} • Cancers: {' '.join(selected_cancers.split('_'))}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"{modality}_gmm_vs_cancer.png"), dpi=300)
    plt.close('all')

    logging.info(f"Saved figures to: {fig_save_folder}")

if __name__ == '__main__':
    main()