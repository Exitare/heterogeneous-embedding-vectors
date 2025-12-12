from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import logging
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
import pandas as pd

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

    # ---------------------------------------------------------
    # Silhouette analysis & intra/inter-cluster distance stats
    # ---------------------------------------------------------

    # (A) Silhouette using TRUE cancer labels (how well cancers separate in embedding space)
    # Encode string labels for sklearn silhouette
    _, cancer_inverse = np.unique(cancer_types, return_inverse=True)
    sil_true_global = silhouette_score(X_data, cancer_inverse, metric='euclidean')

    # (B) Silhouette using GMM cluster labels (how well the fitted clusters separate)
    sil_gmm_global = silhouette_score(X_data, cluster_labels, metric='euclidean')

    # (C) Per-sample silhouettes for TRUE labels, then aggregate by cancer type
    sil_samples_true = silhouette_samples(X_data, cancer_inverse, metric='euclidean')
    per_cancer_sil = (
        pd.DataFrame({"cancer": cancer_types, "silhouette": sil_samples_true})
        .groupby("cancer")["silhouette"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "silhouette_mean", "std": "silhouette_std", "count": "n"})
    )

    # (D) Intra- and inter- cluster distances per cancer (exact mean pairwise distances)
    #     - intra: mean of all pairwise distances within the cancer
    #     - inter: for each other cancer, mean of all cross distances; take the minimum (nearest other cluster)
    records = []
    unique_cancers_list = list(np.unique(cancer_types))

    for c in unique_cancers_list:
        mask_c = (cancer_types == c)
        Xc = X_data[mask_c]
        # Intra distances (upper triangle to avoid double counting & zeros)
        if Xc.shape[0] > 1:
            d_intra = pairwise_distances(Xc, Xc, metric='euclidean')
            tri = np.triu_indices_from(d_intra, k=1)
            intra_mean = d_intra[tri].mean() if tri[0].size > 0 else np.nan
        else:
            intra_mean = np.nan

        # Inter distances: mean distance to each other group's points; take the minimum
        inter_means = []
        for c2 in unique_cancers_list:
            if c2 == c:
                continue
            Xo = X_data[cancer_types == c2]
            if Xc.size > 0 and Xo.size > 0:
                d_inter = pairwise_distances(Xc, Xo, metric='euclidean')
                inter_means.append(d_inter.mean())
        nearest_inter_mean = np.min(inter_means) if len(inter_means) else np.nan

        # Class-level silhouette derived from intra/inter means
        if np.isfinite(intra_mean) and np.isfinite(nearest_inter_mean) and max(intra_mean, nearest_inter_mean) > 0:
            s_class = (nearest_inter_mean - intra_mean) / max(intra_mean, nearest_inter_mean)
        else:
            s_class = np.nan

        records.append({
            "cancer": c,
            "intra_mean_dist": intra_mean,
            "nearest_inter_mean_dist": nearest_inter_mean,
            "silhouette_class_from_means": s_class,
            "n_samples": int(mask_c.sum())
        })

    per_cancer_dist = pd.DataFrame(records)

    # Merge per-cancer silhouette (from samples) with distance stats
    per_cancer_summary = per_cancer_sil.merge(per_cancer_dist, on="cancer", how="outer")

    # Print concise summary
    logging.info(f"Global silhouette (TRUE cancer labels): {sil_true_global:.4f}")
    logging.info(f"Global silhouette (GMM cluster labels): {sil_gmm_global:.4f}")
    logging.info("Per-cancer summary:\n" + per_cancer_summary.sort_values("cancer").to_string(index=False))

    # Save to CSV
    metrics_folder = Path(fig_save_folder)  # you already created this
    per_cancer_summary.to_csv(metrics_folder / f"{walk_distance}_{walk_amount}_silhouette_and_distances_by_cancer.csv",
                              index=False)

    # Optional: Save a compact text file with the two global scores
    with open(metrics_folder / f"{walk_distance}_{walk_amount}_global_silhouette.txt", "w") as f:
        f.write(f"Silhouette (TRUE cancer labels): {sil_true_global:.6f}\n")
        f.write(f"Silhouette (GMM cluster labels):  {sil_gmm_global:.6f}\n")


    ################################################################
    ############# Visualization of Clustering Results ################
    ################################################################

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
