import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
import math

fig_save_folder = Path("results", "classifier", "distance_plots")
load_folder = Path("results", "classifier", "embeddings", "annotated_cancer")
results_save_folder = Path("results", "classifier", "distances")

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

    # min_max_scaler = MinMaxScaler()
    # summed_embeddings.iloc[:, :-2] = min_max_scaler.fit_transform(summed_embeddings.iloc[:, :-2])

    fig_save_folder = Path(fig_save_folder, cancers, f"{walk_distance}_{walk_amount}")
    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_dfs = {}
    y_lim = 0
    for cancer in selected_cancers:
        # Select rows where the specific cancer column has a non-zero value
        cancer_rows = summed_embeddings[summed_embeddings["cancer"] == cancer]

        # Drop the cancer columns and Image Text and RNA
        cancer_rows = cancer_rows.drop(columns=["patient_id", "cancer"])

        # assert that no Image, Text or RNA columns are present
        assert "patient_id" not in cancer_rows.columns, "Patient id column is present"
        assert "cancer" not in cancer_rows.columns, "Cancer column is present"

        # Add the selected rows to the dictionary with the cancer type as the key
        cancer_dfs[cancer] = cancer_rows

    intra_distances = {}
    for cancer, df in cancer_dfs.items():
        intra_distance = euclidean_distances(df)
        intra_distance = intra_distance[np.triu_indices_from(intra_distance, k=1)]
        intra_distances[cancer] = intra_distance

    # calculate the distance between all cancer dfs and create a dictionary that allows to track the distance
    inter_distances = {}
    for cancer1, df1 in cancer_dfs.items():
        for cancer2, df2 in cancer_dfs.items():
            if cancer1 == cancer2:
                continue
            inter_distance = euclidean_distances(df1, df2).flatten()
            inter_distances[(cancer1, cancer2)] = inter_distance

    # Output the average distances
    for cancer, distances in intra_distances.items():
        if y_lim < np.mean(distances):
            y_lim = np.mean(distances)
        print(f"Average intra-cancer distance for {cancer}: {np.mean(distances)}")

    # output the average inter distances
    for (cancer1, cancer2), distances in inter_distances.items():
        if y_lim < np.mean(distances):
            y_lim = np.mean(distances)
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
    cleaned_inter_df = inter_df[inter_df['Cancer'].apply(lambda x: x.split('-')[0] < x.split('-')[1])]

    # Plotting the distances both inter and intra, create two axes
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # create hist plot for intra-cluster distances with a hue
    sns.histplot(intra_df, x='Distance', hue='Cancer', kde=True, ax=ax[0])
    sns.histplot(cleaned_inter_df, x='Distance', kde=True, hue="Cancer", ax=ax[1])
    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')
    plt.xlabel('Cancer Pair')
    plt.ylabel('Euclidean Distance')
    plt.savefig(Path(fig_save_folder, "euclidean_hist_plot.png"), dpi=150)
    plt.close('all')

    # create bar plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=intra_df, x='Cancer', y='Distance', ax=ax[0])
    sns.barplot(data=cleaned_inter_df, x='Cancer', y='Distance', ax=ax[1])
    ax[0].set_title('Intra-cluster Distances')
    ax[1].set_title('Inter-cluster Distances')
    plt.xlabel('Cancer Pair')
    plt.ylabel('Euclidean Distance')
    # rotate x axis
    plt.xticks(rotation=45)
    ax[0].set_ylim(0, 21)
    ax[1].set_ylim(0, 21)

    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "cancer_euclidean_bar_plot.png"), dpi=150)
    plt.close('all')

    cleaned_combined_df = pd.concat([intra_df, cleaned_inter_df], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.barplot(data=cleaned_combined_df, x='Cancer', y='Distance', hue="Type", ax=ax)
    ax.set_title('Euclidean Distance for cancer pairs')
    plt.xlabel('Cancer Pair')
    ax.set_ylabel('Distance')
    # set y scale
    ax.set_ylim(0, 120)
    # rotate x axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "combined_euclidean_bar_plot.png"), dpi=150)
    plt.close('all')

    combined_df = pd.concat([intra_df, inter_df], axis=0)
    primary_cancers = {}
    # create a dict where each key is a primary cancer and each value is the list of matching cancers
    for cancer in combined_df["Cancer"].unique():
        primary_cancer = cancer.split("-")[0]
        if primary_cancer not in primary_cancers:
            primary_cancers[primary_cancer] = [combined_df[combined_df["Cancer"].str.startswith(primary_cancer)]]
        primary_cancers[primary_cancer].append(
            combined_df[combined_df["Cancer"].str.startswith(primary_cancer)])

    # for each key create a panel plot in a new figure
    fig, axs = plt.subplots(len(primary_cancers), 1, figsize=(10, 10))
    for i, (primary_cancer, df) in enumerate(primary_cancers.items()):
        df = pd.concat(df)
        sns.barplot(data=df, x='Cancer', y='Distance', hue="Type", ax=axs[i])
        axs[i].set_title(f'Euclidean Distance for cancer pairs with primary cancer {primary_cancer}')
        axs[i].set_ylabel('Distance')
        # set y scale
        axs[i].set_ylim(0, 120)
        # rotate x axis
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "primary_cancer_euclidean_bar_plot.png"), dpi=150)
    plt.close('all')

    # Calculate the number of rows needed based on the number of primary cancers and 2 columns
    n_cancers = len(primary_cancers)
    n_rows = math.ceil(n_cancers / 2)  # Use math.ceil to round up if there's an odd number of cancers

    # Create subplots with 2 columns and n_rows rows
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 10))

    # Flatten the axes array for easy indexing
    axs = axs.flatten()

    for i, (primary_cancer, df) in enumerate(primary_cancers.items()):
        df = pd.concat(df)
        sns.barplot(data=df, x='Cancer', y='Distance', hue="Type", ax=axs[i])
        axs[i].set_title(f'Euclidean Distance for cancer pairs with primary cancer {primary_cancer}')
        axs[i].set_ylabel('Distance')
        # Set y scale
        axs[i].set_ylim(0, 120)
        # Rotate x axis labels
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

    # Hide any empty subplots if the number of plots is odd
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "primary_cancer_euclidean_bar_plot_cols.png"), dpi=150)
    plt.close('all')
