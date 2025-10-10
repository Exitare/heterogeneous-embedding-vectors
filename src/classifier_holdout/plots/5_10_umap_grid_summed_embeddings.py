import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
import umap
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FIG_ROOT = Path("figures", "classifier_holdout")
SUMMED_ROOT = Path("results", "classifier_holdout", "summed_embeddings")

PALETTE = {
    "BRCA": "red",
    "BLCA": "blue",
    "LUAD": "green",
    "STAD": "purple",
    "THCA": "orange",
    "COAD": "yellow",
}

def load_all_iterations(cancers_key: str, modalities_key: str, wd: int, wa: int) -> tuple[np.ndarray, list[str]]:
    """
    Load and concatenate X, y from ALL iterations under:
      results/classifier_holdout/summed_embeddings/{cancers}/{modalities}/{wd}_{wa}/{iteration}/summed_embeddings.h5
    Returns (X_all, y_all_decoded)
    """
    base = Path(SUMMED_ROOT, cancers_key, modalities_key, f"{wd}_{wa}")
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    X_parts = []
    y_parts = []
    for it_dir in sorted(base.iterdir()):
        if not it_dir.is_dir():
            continue
        h5p = Path(it_dir, "summed_embeddings.h5")
        if not h5p.exists():
            continue
        try:
            with h5py.File(h5p, "r") as f:
                X = f["X"][:]
                raw_y = f["y"][:]
                y = [
                    lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab)
                    for lab in raw_y
                ]
            if X.size == 0 or len(y) == 0:
                logging.warning(f"No data in {h5p}; skipping.")
                continue
            X_parts.append(X)
            y_parts.extend(y)
        except Exception as e:
            logging.warning(f"Failed to read {h5p}: {e}")

    if not X_parts:
        raise FileNotFoundError(f"No summed_embeddings.h5 found under {base}/**/")

    X_all = np.vstack(X_parts)
    return X_all, y_parts


if __name__ == '__main__':
    parser = ArgumentParser(description='Embedding UMAP')
    parser.add_argument("--selected_cancers", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--modalities", "-m", nargs='+', required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modalities to include (space separated).")
    parser.add_argument("--distances", "-d", nargs='+', required=False, type=int,
                        default=[3, 5],
                        help="Range as two ints: min_distance max_distance (inclusive).")
    args = parser.parse_args()

    selected_cancers = args.selected_cancers
    selected_modalities = args.modalities
    distances = args.distances
    if len(distances) != 2:
        raise ValueError("--distances expects exactly two integers: min max")
    min_distance, max_distance = distances
    logging.info(f"Using min_distance: {min_distance} and max_distance: {max_distance}")

    walk_distances = list(range(min_distance, max_distance + 1))
    amount_of_walks = list(range(min_distance, max_distance + 1))

    cancers_key = "_".join(selected_cancers)
    modalities_key = "_".join(sorted(selected_modalities))

    fig_save_folder = Path(FIG_ROOT, cancers_key, modalities_key, "clustering", f"{walk_distances[-1]}_{amount_of_walks[-1]}")
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    # 3x3 combined figure across all (wd, wa)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    plot_idx = 0
    logging.info("Creating combined UMAP figure...")

    for wd in walk_distances:
        for wa in amount_of_walks:
            try:
                X, y = load_all_iterations(cancers_key, modalities_key, wd, wa)
            except FileNotFoundError as e:
                logging.warning(str(e))
                continue

            if X.size == 0 or len(y) == 0:
                logging.warning(f"No data for wd={wd}, wa={wa}; skipping.")
                continue

            reducer = umap.UMAP()
            emb2d = reducer.fit_transform(X)
            df_plot = pd.DataFrame(emb2d, columns=['UMAP1', 'UMAP2'])
            df_plot['cancer'] = y

            sns.scatterplot(
                x='UMAP1', y='UMAP2', hue='cancer', data=df_plot, s=25,
                palette=PALETTE, ax=axes[plot_idx], hue_order=[c for c in selected_cancers if c in set(y)]
            )
            axes[plot_idx].set_title(f'WD: {wd}, AW: {wa}')
            axes[plot_idx].get_legend().remove()
            plot_idx += 1
            if plot_idx >= len(axes):
                break
        if plot_idx >= len(axes):
            break

    plt.tight_layout()
    axes[-1].legend(title='Cancer', loc='lower left')
    plt.suptitle('UMAP Visualization of Summed Embeddings', y=1.02)
    plt.savefig(Path(fig_save_folder, 'combined_summed_embeddings_umap.png'), dpi=150)
    plt.close()

    # Individual UMAP per (wd, wa)
    for wd in walk_distances:
        for wa in amount_of_walks:
            try:
                X, y = load_all_iterations(cancers_key, modalities_key, wd, wa)
            except FileNotFoundError as e:
                logging.warning(str(e))
                continue

            if X.size == 0 or len(y) == 0:
                logging.warning(f"No data for wd={wd}, wa={wa}; skipping.")
                continue

            reducer = umap.UMAP()
            emb2d = reducer.fit_transform(X)
            df_plot = pd.DataFrame(emb2d, columns=['UMAP1', 'UMAP2'])
            df_plot['cancer'] = y

            plt.figure(figsize=(10, 10))
            sns.scatterplot(
                x='UMAP1', y='UMAP2', hue='cancer', data=df_plot, s=25,
                palette=PALETTE, hue_order=[c for c in selected_cancers if c in set(y)]
            )
            plt.title(f'UMAP of Summed Embeddings (WD: {wd}, AW: {wa})')
            plt.legend(title='Cancer', loc='lower left')
            plt.tight_layout()
            plt.savefig(Path(fig_save_folder, f'{wd}_{wa}_umap.png'), dpi=150)
            plt.close()