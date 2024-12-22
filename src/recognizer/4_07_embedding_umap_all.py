import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap

fig_save_folder = Path("results", "recognizer", "umap")

if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--cancer", "-c", nargs='+', required=True)
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[2, 3, 4, 5, 6, 7, 7, 8, 9], default=3)
    args = parser.parse_args()

    selected_cancers = args.cancer
    walk_distance: int = args.walk_distance

    cancers = "_".join(selected_cancers)

    fig_save_folder = Path(fig_save_folder, cancers)

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    summed_embeddings = pd.read_csv(
        Path("results", "recognizer", "summed_embeddings", "multi", cancers, f"{walk_distance}_embeddings.csv"))

    cancer_dfs = {}
    for cancer in selected_cancers:
        # Select rows where the specific cancer column has a non-zero value
        cancer_rows = summed_embeddings[summed_embeddings[cancer] > 0]

        # Drop the cancer columns and Image Text and RNA
        cancer_rows = cancer_rows.drop(columns=["Image", "Text", "RNA", "Mutation"] + selected_cancers)
        cancer_rows["cancer"] = cancer

        # assert that no Nan values are present
        assert not cancer_rows.isnull().values.any(), "Nan values are present"

        # assert that no Image, Text or RNA columns are present
        assert "Image" not in cancer_rows.columns, "Image column is present"
        assert "Text" not in cancer_rows.columns, "Text column is present"
        assert "RNA" not in cancer_rows.columns, "RNA column is present"

        # Add the selected rows to the dictionary with the cancer type as the key
        cancer_dfs[cancer] = cancer_rows

    loaded_embeddings = pd.concat(cancer_dfs.values(), ignore_index=True)

    # Apply KMeans
    print(f"Using {len(selected_cancers)} clusters...")
    kmeans = KMeans(n_clusters=len(selected_cancers), random_state=42)
    loaded_embeddings['cluster'] = kmeans.fit_predict(loaded_embeddings.drop('cancer', axis=1))

    # Step 1: Calculate the count of each (cluster, cancer) pair
    cluster_cancer_counts = loaded_embeddings.groupby(['cluster', 'cancer']).size().reset_index(name='count')

    # Step 2: For each cluster, determine the most frequent cancer type (dominant class)
    dominant_cancer_per_cluster = cluster_cancer_counts.loc[cluster_cancer_counts.groupby('cluster')['count'].idxmax()]

    # Step 3: Calculate the number of correctly assigned samples per cluster
    correct_assignments = dominant_cancer_per_cluster['count'].sum()

    # Step 4: Calculate total number of samples
    total_samples = len(loaded_embeddings)

    # Step 5: Calculate overall purity
    purity = correct_assignments / total_samples
    print(f'Purity: {purity:.2f}')

    # Ensure only numeric data is passed to UMAP
    numeric_df = loaded_embeddings.drop(columns=['cluster', 'cancer'])

    # Apply UMAP to reduce dimensions to 2D for visualization
    umap_reducer = umap.UMAP(random_state=42)
    df_umap = umap_reducer.fit_transform(numeric_df)

    # Convert to DataFrame for easier plotting
    df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
    df_plot['cancer'] = loaded_embeddings['cancer']
    df_plot['cluster'] = loaded_embeddings['cluster']

    print(df_plot)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', palette='Set1', data=df_plot, s=25)
    plt.legend(title='Cancer', loc='upper left')

    plt.title('UMAP with Named Clusters')
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f'all_embeddings_{walk_distance}_walks.png'), dpi=150)
    plt.close()
