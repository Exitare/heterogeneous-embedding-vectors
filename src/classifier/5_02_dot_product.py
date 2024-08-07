import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

fig_save_folder = Path("results", "classifier", "distance_plots")


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

    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    fig_save_folder = Path(fig_save_folder, cancers)

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_dfs = {}
    for cancer in selected_cancers:
        cancer_dfs[cancer] = pd.read_csv(
            Path("results", "classifier", "embeddings", "annotated_cancer", cancers,
                 f"{cancer.lower()}_embeddings.csv"))

    # drop the submitter_id and patient column
    for cancer, df in cancer_dfs.items():
        cancer_dfs[cancer] = df.drop(columns=["submitter_id", "patient"])

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
        'Distance': np.concatenate(list(intra_distances.values()))
    })

    # Convert inter_distances to DataFrame
    inter_df = pd.DataFrame({
        'Cancer_Pair': np.repeat([f"{k[0]}-{k[1]}" for k in inter_distances.keys()],
                                 [len(v) for v in inter_distances.values()]),
        'Distance': np.concatenate(list(inter_distances.values()))
    })

    # only keep one of the cancer pair e.g. BRCA-STAD and STAD-BRCA, only keep one
    inter_df = inter_df[inter_df['Cancer_Pair'].apply(lambda x: x.split('-')[0] < x.split('-')[1])]

    # Plotting the distances both inter and intra, create two axes
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # create hist plot for intra-cluster distances with a hue
    sns.histplot(intra_df, x='Distance', hue='Cancer', kde=True, ax=ax[0])
    sns.histplot(inter_df, x='Distance', kde=True, hue="Cancer_Pair", ax=ax[1])

    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')

    plt.xlabel('Dot Product Distance')
    plt.ylabel('Frequency')
    plt.savefig(Path(fig_save_folder, "dot_hist_plot.png"), dpi=150)
    plt.close('all')

    # create bar plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=intra_df, x='Cancer', y='Distance', ax=ax[0])
    sns.barplot(data=inter_df, x='Cancer_Pair', y='Distance', ax=ax[1])
    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')
    plt.xlabel('Cancer Pair')
    plt.ylabel('Dot Product Distance')
    # rotate x axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "dot_bar_plot.png"), dpi=150)
    plt.close('all')
