from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import logging
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

FIG_ROOT = Path("figures", "classifier_holdout")
RES_DIST_ROOT = Path("results", "classifier_holdout", "distances")
SUMMED_ROOT = Path("results", "classifier_holdout", "summed_embeddings")

walk_distances = [3, 4, 5]
walk_amounts = [3, 4, 5]

def dot_product_distance(X, Y=None):
    if Y is None:
        Y = X
    dot_product = np.dot(X, Y.T)
    min_val = dot_product.min()
    max_val = dot_product.max()
    normalized = (dot_product - min_val) / (max_val - min_val + 1e-12)
    return -normalized

def create_polar_line_plot(df, distance_type, ax, color_dict, all_combos):
    df_filtered = df[df['type'] == distance_type]
    group = df_filtered.groupby(['cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    group['combo'] = group.apply(lambda r: f"{r['walk_distance']}_{r['walk_amount']}", axis=1)

    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda r: f"{r['walk_distance']}_{r['walk_amount']}", axis=1)

    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')
    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    ax.set_xticks(angles[:-1])
    pretty = all_combos_df['combo'].apply(lambda x: f"SC: {x.split('_')[0]}  R: {x.split('_')[1]}")
    ax.set_xticklabels(pretty, fontsize=9)

    for cancer in group['cancer'].dropna().unique():
        cancer_data = merged[merged['cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        if not combos:
            continue
        distances = cancer_data['distance'].tolist()
        distances = [0 if np.isnan(d) else d for d in distances]
        distances += distances[:1]
        cancer_angles = [combo_to_angle[c] for c in combos] + [combo_to_angle[combos[0]]]
        ax.plot(cancer_angles, distances, label=cancer, color=color_dict.get(cancer, None), linewidth=2)
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer, None), s=35, edgecolors='w', zorder=5)

    ax.set_title(f"{distance_type.capitalize()} Distances", va='bottom', fontsize=13, fontweight='bold')
    max_distance = merged['distance'].max(skipna=True)
    if pd.notna(max_distance):
        ax.set_ylim(0, float(max_distance) * 1.1)

def create_polar_inter_plot(df, ax, color_dict, all_combos):
    df_filtered = df[df['type'] == 'inter']
    if df_filtered.empty:
        return

    g1 = df_filtered.groupby(['cancer1', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    g1 = g1.rename(columns={'cancer1': 'cancer'})
    g2 = df_filtered.groupby(['cancer2', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    g2 = g2.rename(columns={'cancer2': 'cancer'})
    group = pd.concat([g1, g2], ignore_index=True)
    group = group.groupby(['cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()

    group['combo'] = group.apply(lambda r: f"{r['walk_distance']}_{r['walk_amount']}", axis=1)

    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda r: f"{r['walk_distance']}_{r['walk_amount']}", axis=1)

    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')
    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_combos_df['combo'], fontsize=9)

    for cancer in group['cancer'].dropna().unique():
        cancer_data = merged[merged['cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        if not combos:
            continue
        distances = cancer_data['distance'].tolist()
        distances = [0 if np.isnan(d) else d for d in distances]
        distances += distances[:1]
        cancer_angles = [combo_to_angle[c] for c in combos] + [combo_to_angle[combos[0]]]
        ax.plot(cancer_angles, distances, label=cancer, color=color_dict.get(cancer, None), linewidth=2)
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer, None), s=35, edgecolors='w', zorder=5)

    ax.set_title("Inter-Class Distances", va='bottom', fontsize=13, fontweight='bold')
    max_distance = merged['distance'].max(skipna=True)
    if pd.notna(max_distance):
        ax.set_ylim(0, float(max_distance) * 1.1)

def main_polar_plots(combined_df: pd.DataFrame, file_path: Path):
    cancers = combined_df['cancer'].dropna().unique()
    palette = sns.color_palette("hsv", len(cancers))
    color_dict = dict(zip(cancers, palette))

    all_combos = list(itertools.product(walk_distances, walk_amounts))

    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))
    create_polar_line_plot(combined_df, 'intra', axes[0], color_dict, all_combos)
    create_polar_inter_plot(combined_df, axes[1], color_dict, all_combos)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, title="Cancer Types")

    plt.suptitle("Polar Plots of Intra and Inter Cancer Type Distances", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def load_concat_for_combo(cancers_key: str, modalities_key: str, wd: int, wa: int) -> pd.DataFrame:
    """
    Load and concat *all iterations* for a given (wd, wa) by globbing:
    results/classifier_holdout/summed_embeddings/<cancers>/<modalities>/<wd>_<wa>/<iteration>/summed_embeddings.h5
    """
    root = Path(SUMMED_ROOT, cancers_key, modalities_key)
    if not root.exists():
        logging.warning(f"Missing root: {root}")
        return pd.DataFrame()

    # Glob all iteration H5s for this combo
    h5_paths = sorted(root.glob(f"{wd}_{wa}/*/summed_embeddings.h5"))
    if not h5_paths:
        # Nothing for this combo — just return empty (don’t error)
        return pd.DataFrame()

    frames = []
    for h5_path in h5_paths:
        try:
            with h5py.File(h5_path, "r") as f:
                X = f["X"][:]
                y = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in f["y"][:]]
            df = pd.DataFrame(X)
            df["cancer"] = y
            df["walk_distance"] = wd
            df["walk_amount"] = wa
            # optional: keep iteration for debugging/stratified analysis
            df["iteration"] = h5_path.parent.name
            frames.append(df)
        except Exception as e:
            logging.warning(f"Failed to read {h5_path}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def calculate_intra_inter_distances(summed_embeddings: dict, selected_cancers: list, distance_metric: str = "euclidean"):
    intra_distances = {}
    inter_distances = {}

    for (wd, wa), df in summed_embeddings.items():
        cancer_dfs = {}
        for cancer in selected_cancers:
            cancer_rows = df[df["cancer"] == cancer].copy()
            if cancer_rows.empty:
                continue
            cancer_rows = cancer_rows.drop(columns=["cancer", "walk_distance", "walk_amount", "iteration"], errors="ignore")
            cancer_dfs[cancer] = cancer_rows

        for cancer, df_c in cancer_dfs.items():
            if df_c.empty:
                continue
            if distance_metric == "euclidean":
                d = euclidean_distances(df_c)
            elif distance_metric == "cosine":
                d = cosine_distances(df_c)
            elif distance_metric == "dot_product":
                d = dot_product_distance(df_c)
            else:
                raise ValueError(f"Invalid distance metric: {distance_metric}")
            d = d[np.triu_indices_from(d, k=1)]
            intra_distances[(wd, wa, cancer)] = d

        cancers_list = list(cancer_dfs.keys())
        for i in range(len(cancers_list)):
            for j in range(i + 1, len(cancers_list)):
                c1, c2 = cancers_list[i], cancers_list[j]
                df1, df2 = cancer_dfs[c1], cancer_dfs[c2]
                if df1.empty or df2.empty:
                    continue
                if distance_metric == "euclidean":
                    d = euclidean_distances(df1, df2).flatten()
                elif distance_metric == "cosine":
                    d = cosine_distances(df1, df2).flatten()
                elif distance_metric == "dot_product":
                    d = dot_product_distance(df1, df2).flatten()
                else:
                    raise ValueError(f"Invalid distance metric: {distance_metric}")
                inter_distances[(wd, wa, c1, c2)] = d

    return intra_distances, inter_distances

def convert_to_records(intra_df: dict, inter_df: dict):
    intra_records = []
    for (wd, wa, cancer), distances in intra_df.items():
        for d in distances:
            intra_records.append({
                "type": "intra",
                "walk_distance": wd,
                "walk_amount": wa,
                "cancer": cancer,
                "distance": float(d),
                "combined_cancer": cancer
            })
    intra_df = pd.DataFrame(intra_records)

    inter_records = []
    for (wd, wa, c1, c2), distances in inter_df.items():
        for d in distances:
            inter_records.append({
                "type": "inter",
                "walk_distance": wd,
                "walk_amount": wa,
                "cancer1": c1,
                "cancer2": c2,
                "cancer": c1,
                "distance": float(d),
                "combined_cancer": f"{c1}-{c2}"
            })
    inter_df = pd.DataFrame(inter_records)
    return intra_df, inter_df

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--modalities", "-m", nargs='+', required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modalities to include (space separated).")
    args = parser.parse_args()

    selected_cancers = args.cancer
    selected_modalities = args.modalities

    cancers_key = "_".join(selected_cancers)
    modalities_key = "_".join(sorted(selected_modalities))

    figure_dir = Path(FIG_ROOT, cancers_key, modalities_key, "distances")
    results_dir = Path(RES_DIST_ROOT, cancers_key, modalities_key)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summed_embeddings = {}
    for wd in walk_distances:
        for wa in walk_amounts:
            df = load_concat_for_combo(cancers_key, modalities_key, wd, wa)
            if df.empty:
                logging.info(f"No iterations found for {modalities_key} {wd}_{wa}; skipping combo.")
                continue
            summed_embeddings[(wd, wa)] = df

    if not summed_embeddings:
        raise FileNotFoundError("No summed_embeddings found for any (walk_distance, walk_amount) combo (across iterations).")

    logging.info("Calculating euclidean distances")
    eu_intra, eu_inter = calculate_intra_inter_distances(summed_embeddings, selected_cancers, "euclidean")

    logging.info("Calculating cosine distances")
    co_intra, co_inter = calculate_intra_inter_distances(summed_embeddings, selected_cancers, "cosine")

    logging.info("Calculating dot_product distances")
    dp_intra, dp_inter = calculate_intra_inter_distances(summed_embeddings, selected_cancers, "dot_product")

    logging.info("Converting to dataframes")
    eu_intra_df, eu_inter_df = convert_to_records(eu_intra, eu_inter)
    co_intra_df, co_inter_df = convert_to_records(co_intra, co_inter)
    dp_intra_df, dp_inter_df = convert_to_records(dp_intra, dp_inter)

    eu_combined = pd.concat([eu_intra_df, eu_inter_df], ignore_index=True)
    co_combined = pd.concat([co_intra_df, co_inter_df], ignore_index=True)
    dp_combined = pd.concat([dp_intra_df, dp_inter_df], ignore_index=True)

    eu_path = Path(results_dir, "euclidean_combined_distances.csv")
    co_path = Path(results_dir, "cosine_combined_distances.csv")
    dp_path = Path(results_dir, "dot_product_combined_distances.csv")
    eu_combined.to_csv(eu_path, index=False)
    co_combined.to_csv(co_path, index=False)
    dp_combined.to_csv(dp_path, index=False)

    logging.info("Creating plots...")
    main_polar_plots(eu_combined, Path(figure_dir, "euclidean_polar.png"))
    main_polar_plots(co_combined, Path(figure_dir, "cosine_polar.png"))
    main_polar_plots(dp_combined, Path(figure_dir, "dot_product_polar.png"))