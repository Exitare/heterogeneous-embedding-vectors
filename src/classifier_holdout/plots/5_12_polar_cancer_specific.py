import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser

# Use new classifier_holdout layout
FIG_ROOT = Path("figures", "classifier_holdout")
RES_ROOT = Path("results", "classifier_holdout", "distances")

# Supported ranges
walk_distances = [3, 4, 5, 6]
walk_amounts = [3, 4, 5, 6]


def create_polar_line_plot(df, primary_cancer, ax, color_dict, all_combos, metric="euclidean"):
    group = df.groupby(['combined_cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    group['combo'] = group.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')

    if metric == "dot_product":
        min_distance = merged['distance'].min()
        if pd.notnull(min_distance) and min_distance < 0:
            merged['distance'] = merged['distance'] - min_distance

    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    ax.set_xticks(angles[:-1])
    all_combos_df['combo'] = all_combos_df['combo'].apply(lambda x: f"SC: {x.split('_')[0]}\n R: {x.split('_')[1]}")
    ax.set_xticklabels(all_combos_df['combo'], fontsize=10)

    cancers = group['combined_cancer'].unique()
    for cancer in cancers:
        cancer_data = merged[merged['combined_cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        if not combos:
            continue
        distances = cancer_data['distance'].tolist()
        distances = [d if not np.isnan(d) else 0 for d in distances]
        distances += distances[:1]
        cancer_angles = [combo_to_angle[combo] for combo in combos] + [combo_to_angle[combos[0]]]

        cancer_label = cancer if "-" in cancer else f"{cancer}-{cancer}"
        ax.plot(cancer_angles, distances, label=cancer_label, color=color_dict.get(cancer_label, None), linewidth=2)
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer_label, None), s=50, edgecolors='w', zorder=5)

    title = f"{primary_cancer} - "
    title += "Euclidean Distance" if metric == "euclidean" else "Cosine" if metric == "cosine" else "Dot Product"
    ax.set_title(title, va='bottom', fontsize=14, fontweight='bold')

    max_distance = merged['distance'].max()
    if pd.notnull(max_distance):
        ax.set_ylim(0, float(max_distance) * 1.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), title="Cancer Type")


def main_polar_plots_for_primary_cancers(combined_df: pd.DataFrame, metric: str, fig_dir: Path):
    primary_cancers = {}
    for cancer in combined_df["combined_cancer"].unique():
        primary = cancer.split("-")[0]
        primary_cancers.setdefault(primary, [])
        primary_cancers[primary].append(combined_df[combined_df["combined_cancer"].str.contains(primary)])

    # Build a color palette across all combined_cancer labels to keep colors consistent
    all_combined = sorted(combined_df["combined_cancer"].unique())
    palette = sns.color_palette("hsv", len(all_combined))
    color_dict = dict(zip(all_combined, palette))

    all_combos = [(wd, wa) for wd in walk_distances for wa in walk_amounts]

    for primary_cancer, dfs in primary_cancers.items():
        df = pd.concat(dfs, ignore_index=True)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        create_polar_line_plot(df, primary_cancer, ax, color_dict, all_combos, metric=metric)
        plt.tight_layout()
        plt.savefig(Path(fig_dir, f"{primary_cancer}_{metric}_polar.png"), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=False,
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--modalities", "-m", nargs='+', required=True,
                        choices=["rna", "annotations", "mutations", "images"],
                        help="Modalities to include (space separated).")
    parser.add_argument("--distance_metric", "-dm", type=str, required=True,
                        choices=["euclidean", "cosine", "dot_product"], default="euclidean")
    args = parser.parse_args()

    selected_cancers = args.cancer
    selected_modalities = args.modalities
    distance_metric = args.distance_metric

    cancers_key = "_".join(selected_cancers)
    modalities_key = "_".join(sorted(selected_modalities))

    save_folder = Path(FIG_ROOT, cancers_key, modalities_key, "distances")
    results_load_folder = Path(RES_ROOT, cancers_key, modalities_key)
    save_folder.mkdir(parents=True, exist_ok=True)
    results_load_folder.mkdir(parents=True, exist_ok=True)

    file_name = {
        "euclidean": "euclidean_combined_distances.csv",
        "cosine": "cosine_combined_distances.csv",
        "dot_product": "dot_product_combined_distances.csv"
    }[distance_metric]

    combined_df = pd.read_csv(Path(results_load_folder, file_name))

    main_polar_plots_for_primary_cancers(combined_df, distance_metric, save_folder)