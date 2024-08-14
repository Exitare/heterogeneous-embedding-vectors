import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
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

    # load embeddings
    loaded_cancer_embeddings = {}
    for cancer in selected_cancers:
        try:
            temp_df = pd.read_csv(Path(cancer_embedding_load_folder, f"{cancer.lower()}_embeddings.csv"))
            # remove patient column if exist
            if "Patient" in temp_df.columns:
                temp_df.drop(columns=["Patient"], inplace=True)
            cancer_type = cancer
            temp_df["cancer"] = cancer
            loaded_cancer_embeddings[cancer_type] = temp_df

        except:
            print(f"Could not load {cancer} embedding...")
            raise

    loaded_cancer_embeddings = pd.concat(loaded_cancer_embeddings.values(), axis=0)

    kmeans = KMeans(n_clusters=len(selected_cancers), random_state=42)
    loaded_cancer_embeddings['cluster'] = kmeans.fit_predict(loaded_cancer_embeddings.drop('cancer', axis=1))

    # Determine the majority class in each cluster
    majority_labels = loaded_cancer_embeddings.groupby('cluster')['cancer'].agg(lambda x: x.value_counts().index[0])
    loaded_cancer_embeddings['majority_label'] = loaded_cancer_embeddings['cluster'].map(majority_labels)

    # Reduce to 2D for visualization
    # Apply UMAP to reduce dimensions to 2D for visualization
    umap_reducer = umap.UMAP(random_state=42)
    df_umap = umap_reducer.fit_transform(loaded_cancer_embeddings.drop(['cluster', 'cancer', 'majority_label'], axis=1))

    # Convert to DataFrame for easier plotting
    df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
    df_plot['cluster'] = loaded_cancer_embeddings['cluster']
    df_plot['majority_label'] = loaded_cancer_embeddings['majority_label']

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', palette='Set1', data=df_plot, s=100)

    # Annotate the majority label on the plot
    for i in range(len(df_plot)):
        plt.text(df_plot['UMAP1'][i], df_plot['UMAP2'][i], df_plot['majority_label'][i],
                 fontsize=9, ha='center')

    plt.title('UMAP with Clusters and Majority Labels')
    plt.show()
    plt.savefig(Path(fig_save_folder, 'cancer_embeddings_cluster.png'), dpi=150)
