import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap

load_folder = Path("results", "classifier", "summed_embeddings")
fig_save_folder = Path("results", "classifier", "umap")

if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=True)
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5], default=3)
    args = parser.parse_args()

    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)
    walk_distance = args.walk_distance
    amount_of_walks = args.amount_of_walks

    fig_save_folder = Path(fig_save_folder, cancers)

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    load_folder = Path(load_folder, cancers)
    load_folder = Path(load_folder, f"{walk_distance}_{amount_of_walks}")
    loaded_cancer_embeddings = pd.read_csv(Path(load_folder, "summed_embeddings.csv"))

    # assert that all selected cancers are in the cancer column of the loaded_cancer_embeddings
    assert all([cancer in loaded_cancer_embeddings["cancer"].unique() for cancer in
                selected_cancers]), "All selected cancers should be in the cancer column"

    # save cancer name
    available_cancers = loaded_cancer_embeddings["cancer"]

    loaded_cancer_embeddings.drop(columns=["patient_id", "cancer"], inplace=True)
    loaded_cancer_embeddings.reset_index(drop=True, inplace=True)
    # remove all rows that have nan values
    loaded_cancer_embeddings.dropna(inplace=True)

    # assert that submitter id, patient is not in columns
    assert "cancer" not in loaded_cancer_embeddings.columns, "submitter_id should not be in the columns"
    assert "patient_id" not in loaded_cancer_embeddings.columns, "patient should not be in the columns"

    # Apply UMAP to reduce dimensions to 2D for visualization
    umap_reducer = umap.UMAP(random_state=42)
    df_umap = umap_reducer.fit_transform(loaded_cancer_embeddings)

    # Convert to DataFrame for easier plotting
    df_plot = pd.DataFrame(df_umap, columns=['UMAP1', 'UMAP2'])
    df_plot['cancer'] = available_cancers

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='cancer',
                    palette={"BRCA": "red", "BLCA": "blue", "LUAD": "green", "STAD": "purple", "THCA": "orange",
                             "COAD": "yellow"}, data=df_plot, s=25)
    plt.legend(title='Cancer', loc='upper left')

    plt.title('UMAP Visualization of Summed Embeddings')
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, 'summed_embeddings.png'), dpi=150)
    plt.close()
