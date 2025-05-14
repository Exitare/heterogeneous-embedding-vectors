import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser

# Corrected walk_distances to include 4
walk_distances = [3, 4, 5]
walk_amounts = [3, 4, 5]

results_load_folder = Path("results", "classifier", "distances")
save_folder = Path("figures", "classifier")


def create_polar_line_plot(df, primary_cancer, ax, color_dict, all_combos, metric="euclidean"):
    """
    Creates a polar line plot for intra-class distances for each primary cancer.
    """
    # Group by combined_cancer, walk_distance, walk_amount to compute the average distance
    group = df.groupby(['combined_cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()

    # Create a unique identifier for each combination
    group['combo'] = group.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Create a DataFrame with all possible combinations
    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Merge to ensure all combinations are present, filling missing with NaN
    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')

    # If using dot_product, shift distances so that the minimum value becomes 0.
    if metric == "dot_product":
        min_distance = merged['distance'].min()
        if pd.notnull(min_distance) and min_distance < 0:
            merged['distance'] = merged['distance'] - min_distance

    # Sort by walk_distance and walk_amount
    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    # Assign each combination a unique angle
    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Ensure the plot is circular by appending the first angle at the end
    angles += angles[:1]

    # Create a mapping from combo to angle
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    # Set the labels for each combination
    ax.set_xticks(angles[:-1])
    all_combos_df['combo'] = all_combos_df['combo'].apply(lambda x: f"SC: {x.split('_')[0]} and R: {x.split('_')[1]}")

    # Now, set the tick labels
    ax.set_xticklabels(all_combos_df['combo'], fontsize=10)

    # Plot data for each cancer type within the primary cancer group
    cancers = group['combined_cancer'].unique()
    for cancer in cancers:
        cancer_data = merged[merged['combined_cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        distances = cancer_data['distance'].tolist()

        # Handle missing distances by setting them to 0
        distances = [d if not np.isnan(d) else 0 for d in distances]

        # Append the first distance to close the loop
        distances += distances[:1]
        cancer_angles = [combo_to_angle[combo] for combo in combos] + [combo_to_angle[combos[0]]]

        if "-" not in cancer:
            cancer_label = f"{cancer}-{cancer}"
        else:
            cancer_label = cancer

        # Plot the line
        ax.plot(cancer_angles, distances, label=cancer_label, color=color_dict.get(cancer, None), linewidth=2)

        # Plot the points
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer, None), s=50, edgecolors='w', zorder=5)

    title = f"{primary_cancer} -"

    if distance_metric == "euclidean":
        title += " Euclidean Distance"

    elif distance_metric == "cosine":
        title += " Cosine"

    elif distance_metric == "dot_product":
        title += " Dot Product"

    # Set the title
    ax.set_title(title, va='bottom', fontsize=14, fontweight='bold')

    # Set radial limits with some padding (recompute max after shifting)
    max_distance = merged['distance'].max()
    ax.set_ylim(0, max_distance * 1.1)

    # Add the legend for cancer types
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), title="Cancer Type")


def main_polar_plots_for_primary_cancers(combined_df: pd.DataFrame, metric: str):
    """
    Creates one polar plot for each primary cancer.
    """
    # Create a dictionary where each key is a primary cancer and each value is a list of matching cancers
    primary_cancers = {}
    for cancer in combined_df["combined_cancer"].unique():
        primary_cancer = cancer.split("-")[0]
        if primary_cancer not in primary_cancers:
            primary_cancers[primary_cancer] = [combined_df[combined_df["combined_cancer"].str.contains(primary_cancer)]]
        primary_cancers[primary_cancer].append(
            combined_df[combined_df["combined_cancer"].str.contains(primary_cancer)]
        )

    # Prepare color mapping for cancers
    cancers = primary_cancers.keys()
    num_cancers = len(cancers)
    palette = sns.color_palette("hsv", num_cancers)
    color_dict = dict(zip(cancers, palette))

    # Define all possible combinations of walk_distance and walk_amount
    all_combos = [(wd, wa) for wd in walk_distances for wa in walk_amounts]

    # For each primary cancer, create a polar plot in a new figure
    for primary_cancer, dfs in primary_cancers.items():
        df = pd.concat(dfs)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        create_polar_line_plot(df, primary_cancer, ax, color_dict, all_combos, metric=metric)

        # Save the plot
        plt.tight_layout()
        plt.savefig(Path(save_folder, f"{primary_cancer}_{metric}_polar.png"), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=False,
                        help="The cancer type to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--distance_metric", "-dm", type=str, required=True,
                        help="The distance metric to load.",
                        choices=["euclidean", "cosine", "dot_product"], default="euclidean")
    args = parser.parse_args()
    selected_cancers = args.cancer
    distance_metric = args.distance_metric
    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, "distances")
    results_load_folder = Path(results_load_folder, cancers)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    file_name = "euclidean_combined_distances.csv"
    if distance_metric == "cosine":
        file_name = "cosine_combined_distances.csv"
    elif distance_metric == "dot_product":
        file_name = "dot_product_combined_distances.csv"

    combined_df = pd.read_csv(Path(results_load_folder, file_name))

    # Generate polar plots for each primary cancer
    main_polar_plots_for_primary_cancers(combined_df, distance_metric)
