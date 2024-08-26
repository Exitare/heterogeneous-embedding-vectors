import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap

load_folder = Path("results", "classifier", "embeddings")
cancer_embedding_load_folder = Path(load_folder, "annotated_cancer")
fig_save_folder = Path("results", "classifier", "umap")

if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=True)
    args = parser.parse_args()

    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)

    fig_save_folder = Path(fig_save_folder, cancers)

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_embedding_load_folder = Path(cancer_embedding_load_folder, cancers)

    # Load embeddings
    loaded_cancer_embeddings = {}
    for cancer in selected_cancers:
        try:
            temp_df = pd.read_csv(Path(cancer_embedding_load_folder, f"{cancer.lower()}_embeddings.csv"))
            # Remove patient column if exist
            if "Patient" in temp_df.columns:
                temp_df.drop(columns=["Patient"], inplace=True)
            temp_df["cancer"] = cancer
            loaded_cancer_embeddings[cancer] = temp_df

        except Exception as e:
            print(f"Could not load {cancer} embedding... {e}")
            raise

    loaded_cancer_embeddings = pd.concat(loaded_cancer_embeddings.values(), axis=0)
    # reset index
    loaded_cancer_embeddings.reset_index(drop=True, inplace=True)

    # load annotations
    annotations = pd.read_csv(
        Path("results", "classifier", "embeddings", "annotations", cancers, "embeddings.csv"))
    # reset index
    annotations.reset_index(drop=True, inplace=True)
    mutations = pd.read_csv(Path("results", "classifier", "embeddings", "mutation_embeddings.csv"))
    # reset index
    mutations.reset_index(drop=True, inplace=True)

    # assert that all selected cancers are in the cancer column of the loaded_cancer_embeddings
    assert all([cancer in loaded_cancer_embeddings["cancer"].unique() for cancer in
                selected_cancers]), "All selected cancers should be in the cancer column"

    # combine cancer embeddings, annotations and mutations
    loaded_cancer_embeddings = pd.concat([loaded_cancer_embeddings, annotations, mutations], axis=1)
    loaded_cancer_embeddings.drop(columns=["submitter_id", "patient"], inplace=True)
    loaded_cancer_embeddings.reset_index(drop=True, inplace=True)
    # remove all rows that have nan values
    loaded_cancer_embeddings.dropna(inplace=True)

    # assert that submitter id, patient is not in columns
    assert "submitter_id" not in loaded_cancer_embeddings.columns, "submitter_id should not be in the columns"
    assert "patient" not in loaded_cancer_embeddings.columns, "patient should not be in the columns"

    # Apply KMeans
    print(f"Using {len(selected_cancers)} clusters...")
    kmeans = KMeans(n_clusters=len(selected_cancers), random_state=42)
    loaded_cancer_embeddings['cluster'] = kmeans.fit_predict(loaded_cancer_embeddings.drop('cancer', axis=1))

    # Determine the dominant cancer type in each cluster
    cluster_cancer_counts = loaded_cancer_embeddings.groupby(['cluster', 'cancer']).size().reset_index(name='count')
    dominant_cancer_per_cluster = cluster_cancer_counts.loc[cluster_cancer_counts.groupby('cluster')['count'].idxmax()]

    # Create a mapping from cluster to the dominant cancer type
    cluster_to_cancer = dict(zip(dominant_cancer_per_cluster['cluster'], dominant_cancer_per_cluster['cancer']))

    # Apply the corrected mapping to create a 'cluster_name' column
    loaded_cancer_embeddings['cluster_name'] = loaded_cancer_embeddings['cluster'].map(cluster_to_cancer)

    # Ensure only numeric data is passed to UMAP
    numeric_df = loaded_cancer_embeddings.drop(columns=['cluster', 'cancer', 'cluster_name'])

    # Apply UMAP to reduce dimensions to 2D for visualization
    umap_reducer = umap.UMAP(random_state=42)
    df_umap = umap_reducer.fit_transform(numeric_df)

    # Convert to DataFrame for easier plotting
    df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
    df_plot['cluster_name'] = loaded_cancer_embeddings['cluster_name']
    df_plot['cancer'] = loaded_cancer_embeddings['cancer']

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='cancer', palette='Set1', data=df_plot, s=25)
    plt.legend(title='Cancer', loc='upper left')

    plt.title('UMAP Visualization of Cancer Embeddings')
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, 'all_embeddings.png'), dpi=150)
    plt.close()
