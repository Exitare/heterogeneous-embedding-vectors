import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import h5py
import math

fig_save_folder = Path("figures", "classifier")
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

    with h5py.File(Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}",
                        "summed_embeddings.h5"), "r") as f:
        summed_embeddings = pd.DataFrame(f["X"][:])

        summed_embeddings["cancer"] = f["y"][:]
        # encode the cancer type in utf8
        summed_embeddings["cancer"] = summed_embeddings["cancer"].apply(lambda x: x.decode("utf-8"))

    fig_save_folder = Path(fig_save_folder, cancers, "distances", f"{walk_distance}_{walk_amount}")

    if not fig_save_folder.exists():
        fig_save_folder.mkdir(parents=True)

    cancer_dfs = {}
    for cancer in selected_cancers:
        # Select rows where the specific cancer column has a non-zero value
        cancer_rows = summed_embeddings[summed_embeddings["cancer"] == cancer]

        # Drop the cancer columns and Image Text and RNA
        cancer_rows = cancer_rows.drop(columns=["cancer"])

        # assert that no Image, Text or RNA columns are present
        # assert "patient_id" not in cancer_rows.columns, "Patient id column is present"
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

    combined_df = pd.concat([intra_df, inter_df], axis=0)
    primary_cancers = {}
    # create a dict where each key is a primary cancer and each value is the list of matching cancers
    for cancer in combined_df["Cancer"].unique():
        primary_cancer = cancer.split("-")[0]
        if primary_cancer not in primary_cancers:
            primary_cancers[primary_cancer] = [combined_df[combined_df["Cancer"].str.startswith(primary_cancer)]]
        primary_cancers[primary_cancer].append(
            combined_df[combined_df["Cancer"].str.startswith(primary_cancer)])

    # Calculate the number of rows needed based on the number of primary cancers and 2 columns
    n_cancers = len(primary_cancers)
    n_rows = math.ceil(n_cancers / 2)  # Use math.ceil to round up if there's an odd number of cancers

    y_lim: int = 0
    for (primary_cancer, df) in primary_cancers.items():
        df = pd.concat(df)
        for type in df["Type"].unique():
            print(f"Primary Cancer: {primary_cancer}, Type: {type}, Mean: {df[df['Type'] == type]['Distance'].mean()}")
            if y_lim < df[df['Type'] == type]['Distance'].mean():
                y_lim = df[df['Type'] == type]['Distance'].mean()

    y_lim += 0.5
    print("Y Lim:", y_lim)

    # Create subplots with 2 columns and n_rows rows
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 10))

    # Flatten the axes array for easy indexing
    axs = axs.flatten()

    for i, (primary_cancer, df) in enumerate(primary_cancers.items()):
        df = pd.concat(df)
        sns.barplot(data=df, x='Cancer', y='Distance', hue="Type", ax=axs[i])
        axs[i].set_title(f'Cosine Distance for {primary_cancer}')
        axs[i].set_ylabel('Distance')
        axs[i].set_ylim(0, y_lim)
        # Rotate x axis labels
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
        # put legend to left bottom
        axs[i].legend(loc='lower left')

    # Hide any empty subplots if the number of plots is odd
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "cosine_per_cancer.png"), dpi=150)
    plt.close('all')
