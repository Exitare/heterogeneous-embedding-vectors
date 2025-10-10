import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
import math
import h5py
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIG_ROOT = Path("figures", "classifier_holdout")
SUMMED_ROOT = Path("results", "classifier_holdout", "summed_embeddings")

def load_all_iterations(cancers_key: str, modalities_key: str, walk_distance: int, walk_amount: int) -> pd.DataFrame:
    """
    Reads summed_embeddings.h5 from every iteration folder and concatenates.
    Path pattern:
      results/classifier_holdout/summed_embeddings/{cancers}/{modalities}/{walk_distance}_{walk_amount}/{iteration}/summed_embeddings.h5
    """
    base = Path(SUMMED_ROOT, cancers_key, modalities_key, f"{walk_distance}_{walk_amount}")
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    frames = []
    for iter_dir in sorted(base.iterdir()):
        if not iter_dir.is_dir():
            continue
        h5_path = Path(iter_dir, "summed_embeddings.h5")
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                X = f["X"][:]
                y = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else lab for lab in f["y"][:]]
            df = pd.DataFrame(X)
            df["cancer"] = y
            df["iteration"] = iter_dir.name
            frames.append(df)
        except Exception as e:
            logging.warning(f"Failed to read {h5_path}: {e}")

    if not frames:
        raise FileNotFoundError(f"No summed_embeddings.h5 found under {base}/**/")

    return pd.concat(frames, ignore_index=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="Cancer types (space separated).")
    parser.add_argument("--modalities", "-m", nargs='+', required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modalities to include (space separated).")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, choices=[3, 4, 5, 6],
                        help="Walk distance.")
    parser.add_argument("--walk_amount", "-a", type=int, required=True, choices=[3, 4, 5, 6],
                        help="Walk amount.")

    args = parser.parse_args()

    selected_cancers = args.cancer
    selected_modalities = args.modalities
    walk_distance: int = args.walk_distance
    walk_amount: int = args.walk_amount

    if len(selected_cancers) == 1 and " " in selected_cancers[0]:
        selected_cancers = selected_cancers[0].split()

    cancers_key = "_".join(selected_cancers)
    modalities_key = "_".join(sorted(selected_modalities))

    summed_embeddings = load_all_iterations(
        cancers_key=cancers_key,
        modalities_key=modalities_key,
        walk_distance=walk_distance,
        walk_amount=walk_amount
    )

    fig_save_folder = Path(FIG_ROOT, cancers_key, modalities_key, "distances", f"{walk_distance}_{walk_amount}")
    fig_save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Using save folder: {fig_save_folder}")

    cancer_dfs = {}
    for cancer in selected_cancers:
        cancer_rows = summed_embeddings[summed_embeddings["cancer"] == cancer]
        if cancer_rows.empty:
            logging.warning(f"No rows for cancer {cancer}; skipping.")
            continue
        cancer_rows = cancer_rows.drop(columns=["cancer", "iteration"], errors="ignore")
        assert "cancer" not in cancer_rows.columns
        cancer_dfs[cancer] = cancer_rows

    if not cancer_dfs:
        raise ValueError("No valid cancer-specific dataframes were created.")

    intra_distances = {}
    for cancer, df in cancer_dfs.items():
        dmat = euclidean_distances(df)
        intra = dmat[np.triu_indices_from(dmat, k=1)]
        intra_distances[cancer] = intra

    inter_distances = {}
    for c1, df1 in cancer_dfs.items():
        for c2, df2 in cancer_dfs.items():
            if c1 == c2:
                continue
            inter = euclidean_distances(df1, df2).flatten()
            inter_distances[(c1, c2)] = inter

    y_lim = 0.0
    for cancer, distances in intra_distances.items():
        y_lim = max(y_lim, float(np.mean(distances)))
        logging.info(f"Average intra-cancer distance for {cancer}: {np.mean(distances):.4f}")

    for (c1, c2), distances in inter_distances.items():
        y_lim = max(y_lim, float(np.mean(distances)))
        logging.info(f"Average inter-cancer distance for {c1} vs {c2}: {np.mean(distances):.4f}")

    intra_df = pd.DataFrame({
        'Cancer': np.repeat(list(intra_distances.keys()), [len(d) for d in intra_distances.values()]),
        'Distance': np.concatenate(list(intra_distances.values())),
        'Type': 'Intra Cluster'
    })

    inter_df = pd.DataFrame({
        'Cancer': np.repeat([f"{k[0]}-{k[1]}" for k in inter_distances.keys()],
                            [len(v) for v in inter_distances.values()]),
        'Distance': np.concatenate(list(inter_distances.values())),
        'Type': 'Inter Cluster'
    })

    combined_df = pd.concat([intra_df, inter_df], axis=0, ignore_index=True)

    primary_cancers = {}
    for label in combined_df["Cancer"].unique():
        primary = label.split("-")[0]
        primary_cancers.setdefault(primary, [])
        primary_cancers[primary].append(combined_df[combined_df["Cancer"].str.startswith(primary)])

    n_cancers = len(primary_cancers)
    n_rows = math.ceil(n_cancers / 2)

    for (primary, df_list) in primary_cancers.items():
        df = pd.concat(df_list)
        for t in df["Type"].unique():
            y_lim = max(y_lim, float(df[df['Type'] == t]['Distance'].mean()))

    y_lim = y_lim + 10

    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, (primary, df_list) in enumerate(primary_cancers.items()):
        df = pd.concat(df_list)
        sns.barplot(data=df, x='Cancer', y='Distance', hue="Type", ax=axs[i])
        axs[i].set_title(f'Euclidean Distance for {primary}')
        axs[i].set_ylabel('Distance')
        axs[i].set_ylim(0, y_lim)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
        axs[i].legend(loc='lower left')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(Path(fig_save_folder, "euclidean_per_cancer.png"), dpi=150)
    plt.close('all')