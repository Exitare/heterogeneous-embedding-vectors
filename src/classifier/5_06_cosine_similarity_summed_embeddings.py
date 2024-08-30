import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

fig_save_folder = Path("results", "classifier", "distance_plots")
load_folder = Path("results", "classifier", "embeddings", "annotated_cancer")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="The cancer type to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.", choices=[3, 4, 5])
    parser.add_argument("--walk_amount", "-a", type=int, required=True, help="The walk amount.", choices=[3, 4, 5])

    args = parser.parse_args()

    selected_cancers = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.walk_amount

    cancers = "_".join(selected_cancers)

    summed_embeddings = pd.read_csv(
        Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}",
             "summed_embeddings.csv"))

    fig_save_folder = Path(fig_save_folder, cancers, f"{walk_distance}_{walk_amount}")

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_dfs = {}
    for cancer in selected_cancers:
        # Select rows where the specific cancer column has a non-zero value
        cancer_rows = summed_embeddings[summed_embeddings["cancer"] == cancer]

        # Drop the cancer columns and Image Text and RNA
        cancer_rows = cancer_rows.drop(columns=["patient_id","cancer"])

        # assert that no Image, Text or RNA columns are present
        assert "patient_id" not in cancer_rows.columns, "Patient id column is present"
        assert "cancer" not in cancer_rows.columns, "Cancer column is present"

        # Add the selected rows to the dictionary with the cancer type as the key
        cancer_dfs[cancer] = cancer_rows

    intra_distances = {}
    for cancer, df in cancer_dfs.items():
        intra_distance = cosine_distances(df)
        intra_distance = intra_distance[np.triu_indices_from(intra_distance, k=1)]
        intra_distances[cancer] = intra_distance

    # calculate the distance between all cancer dfs and create a dictionary that allows to track the distance
    inter_distances = {}
    for cancer1, df1 in cancer_dfs.items():
        for cancer2, df2 in cancer_dfs.items():
            if cancer1 == cancer2:
                continue
            inter_distance = cosine_distances(df1, df2).flatten()
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
        "Type": "Intra Cluster"
    })

    # Convert inter_distances to DataFrame
    inter_df = pd.DataFrame({
        'Cancer': np.repeat([f"{k[0]}-{k[1]}" for k in inter_distances.keys()],
                            [len(v) for v in inter_distances.values()]),
        'Distance': np.concatenate(list(inter_distances.values())),
        "Type": "Inter Cluster"
    })

    # only keep one of the cancer pair e.g. BRCA-STAD and STAD-BRCA, only keep one
    inter_df = inter_df[inter_df['Cancer'].apply(lambda x: x.split('-')[0] < x.split('-')[1])]

    # Plotting the distances both inter and intra, create two axis
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # create hist plot for intra cluster distances with a hue
    sns.histplot(intra_df, x='Distance', hue='Cancer', kde=True, ax=ax[0])
    sns.histplot(inter_df, x='Distance', kde=True, hue="Cancer", ax=ax[1])

    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')

    # sns.histplot(inter_cluster_distances, kde=True, color='red', label='Inter-cluster Distances')

    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.savefig(Path(fig_save_folder, "cosine_hist_plot.png"), dpi=150)
    plt.close('all')

    # create bar plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(x='Cancer', y='Distance', data=intra_df, ax=ax[0])
    sns.barplot(x='Cancer', y='Distance', data=inter_df, ax=ax[1])
    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')
    plt.xlabel('Cancer Pair')
    plt.ylabel('Cosine Similarity')

    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    # rotate x labels
    for ax_ in ax:
        for label in ax_.get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "cosine_bar_plot.png"), dpi=150)
    plt.close('all')

    combined_df = pd.concat([intra_df, inter_df], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.barplot(data=combined_df, x='Cancer', y='Distance', hue="Type", ax=ax)
    ax.set_title('Cosine Similarity for cancer pairs')
    plt.xlabel('Cancer Pair')
    ax.set_ylabel('Similarity')
    # set y scale
    ax.set_ylim(0, 1)
    # rotate x axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, f"combined_cosine_bar_plot.png"), dpi=150)
    plt.close('all')
