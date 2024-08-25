import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

fig_save_folder = Path("results", "recognizer", "distance_plots")


def dot_product_distance(X, Y=None):
    """
    Compute the dot product distance between rows of X and rows of Y.
    If Y is not provided, compute the dot product distance between rows of X.
    """
    if Y is None:
        Y = X
    return -np.dot(X, Y.T)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="The cancer type to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[2, 3, 4, 5, 6, 7, 7, 8, 9], default=3)
    args = parser.parse_args()

    selected_cancers = args.cancer
    walk_distance: int = args.walk_distance

    cancers = "_".join(selected_cancers)

    summed_embeddings = pd.read_csv(
        Path("results", "recognizer", "summed_embeddings", "multi", cancers, f"{walk_distance}_embeddings.csv"))

    fig_save_folder = Path(fig_save_folder, cancers,
                           f"{walk_distance}_walk_distance_{len(summed_embeddings)}_walk_amount")

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_dfs = {}
    for cancer in selected_cancers:
        # Select rows where the specific cancer column has a non-zero value
        cancer_rows = summed_embeddings[summed_embeddings[cancer] > 0]

        # Drop the cancer columns and Image Text and RNA
        cancer_rows = cancer_rows.drop(columns=["Image", "Text", "RNA"] + selected_cancers)

        # assert that no Image, Text or RNA columns are present
        assert "Image" not in cancer_rows.columns, "Image column is present"
        assert "Text" not in cancer_rows.columns, "Text column is present"
        assert "RNA" not in cancer_rows.columns, "RNA column is present"

        # assert that no columns with the selected cancers are present
        assert all([cancer not in cancer_rows.columns for cancer in selected_cancers]), "Cancer columns are present"

        # Add the selected rows to the dictionary with the cancer type as the key
        cancer_dfs[cancer] = cancer_rows

    intra_distances = {}
    for cancer, df in cancer_dfs.items():
        intra_distance = dot_product_distance(df)
        intra_distance = intra_distance[np.triu_indices_from(intra_distance, k=1)]
        intra_distances[cancer] = intra_distance

    # calculate the distance between all cancer dfs and create a dictionary that allows to track the distance
    inter_distances = {}
    for cancer1, df1 in cancer_dfs.items():
        for cancer2, df2 in cancer_dfs.items():
            if cancer1 == cancer2:
                continue
            inter_distance = dot_product_distance(df1, df2).flatten()
            inter_distances[(cancer1, cancer2)] = inter_distance

    # Output the average distances
    for cancer, distances in intra_distances.items():
        print(f"Average intra-cancer distance for {cancer}: {np.mean(distances)}")

    # output the average inter distances
    for (cancer1, cancer2), distances in inter_distances.items():
        print(f"Average inter-cancer distance for {cancer1} and {cancer2}: {np.mean(distances)}")

    # Convert intra_distances to DataFrame
    intra_df = pd.DataFrame({
        'Cancer': np.repeat(list(intra_distances.keys()), [len(d) for d in intra_distances.values()]),
        'Distance': np.concatenate(list(intra_distances.values())),
        'Type': 'Intra Cluster'
    })

    # Convert inter_distances to DataFrame
    inter_df = pd.DataFrame({
        'Cancer': np.repeat([f"{k[0]}-{k[1]}" for k in inter_distances.keys()],
                                 [len(v) for v in inter_distances.values()]),
        'Distance': np.concatenate(list(inter_distances.values())),
        'Type': 'Inter Cluster'
    })

    # only keep one of the cancer pair e.g. BRCA-STAD and STAD-BRCA, only keep one
    inter_df = inter_df[inter_df['Cancer'].apply(lambda x: x.split('-')[0] < x.split('-')[1])]

    combined_df = pd.concat([intra_df, inter_df], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.barplot(data=combined_df, x='Cancer', y='Distance', hue="Type", ax=ax)
    ax.set_title('Dot Product for cancer pairs')
    plt.xlabel('Cancer Pair')
    ax.set_ylabel('Distance')
    # set y scale
    ax.set_ylim(-750, 0)
    # rotate x axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "all_combined_dot_product_bar_plot.png"), dpi=150)
    plt.close('all')
