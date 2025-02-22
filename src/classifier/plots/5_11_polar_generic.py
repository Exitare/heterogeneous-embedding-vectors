from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py

# Corrected walk_distances to include 4
walk_distances = [3, 4, 5]
walk_amounts = [3, 4, 5]

figure_save_folder = Path("figures", "classifier")
results_save_folder = Path("results", "classifier", "distances")


def dot_product_distance(X, Y=None):
    """
    Compute the dot product distance between rows of X and rows of Y.
    If Y is not provided, compute the dot product distance between rows of X.
    Normalize to make values non-negative for visualization.
    """
    if Y is None:
        Y = X
    dot_product = np.dot(X, Y.T)
    # Normalize to [0, 1] range for better visualization
    min_val = dot_product.min()
    max_val = dot_product.max()
    normalized = (dot_product - min_val) / (max_val - min_val)
    return -normalized  # Negate to align with distance interpretation


def create_polar_line_plot(df, distance_type, ax, color_dict, all_combos):
    """
    Creates a polar line plot for intra-class distances.
    """
    # Filter the DataFrame based on the distance type
    df_filtered = df[df['type'] == distance_type]

    # Group by cancer, walk_distance, walk_amount to compute the average distance
    group = df_filtered.groupby(['cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()

    # Create a unique identifier for each combination
    group['combo'] = group.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Create a DataFrame with all possible combinations
    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Merge to ensure all combinations are present, filling missing with NaN
    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')

    # Sort by walk_distance and walk_amount
    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    # Assign each combination a unique angle
    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Ensure the plot is circular by appending the first value at the end
    angles += angles[:1]

    # Create a mapping from combo to angle
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    # Set the labels for each combination
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_combos_df['combo'], fontsize=10)

    # Iterate over each cancer type and plot its distances
    cancers = group['cancer'].unique()
    for cancer in cancers:
        cancer_data = merged[merged['cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        distances = cancer_data['distance'].tolist()

        # Handle missing distances by setting them to 0
        distances = [d if not np.isnan(d) else 0 for d in distances]

        # Append the first distance to close the loop
        distances += distances[:1]
        cancer_angles = [combo_to_angle[combo] for combo in combos] + [combo_to_angle[combos[0]]]

        # Plot the line
        ax.plot(cancer_angles, distances, label=cancer, color=color_dict.get(cancer, None), linewidth=2)

        # Plot the points
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer, None), s=50, edgecolors='w', zorder=5)

    # Set the title
    ax.set_title(f"{distance_type.capitalize()} Distances", va='bottom', fontsize=14, fontweight='bold')

    # Optional: Set radial limits with some padding
    max_distance = merged['distance'].max()
    ax.set_ylim(0, max_distance * 1.1)


def create_polar_inter_plot(df, ax, color_dict, all_combos):
    """
    Creates a polar line plot for inter-class distances.
    """
    # Filter for inter distances
    df_filtered = df[df['type'] == 'inter']

    # Check if 'cancer1' and 'cancer2' columns exist
    if 'cancer1' not in df_filtered.columns or 'cancer2' not in df_filtered.columns:
        print("Error: 'inter' type DataFrame must contain 'cancer1' and 'cancer2' columns.")
        return

    # Calculate average inter-class distance per cancer type per combination
    # For each (walk_distance, walk_amount), and for each cancer1, compute average distance to all cancer2 != cancer1

    # Group by 'cancer1', 'walk_distance', 'walk_amount' and compute mean distance
    group1 = df_filtered.groupby(['cancer1', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    group1.rename(columns={'cancer1': 'cancer'}, inplace=True)

    # Similarly, group by 'cancer2', 'walk_distance', 'walk_amount' and compute mean distance
    group2 = df_filtered.groupby(['cancer2', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()
    group2.rename(columns={'cancer2': 'cancer'}, inplace=True)

    # Concatenate both groups to get per cancer type average inter-class distance
    group = pd.concat([group1, group2], ignore_index=True)

    # Now, group again by 'cancer', 'walk_distance', 'walk_amount' to compute average if needed
    group = group.groupby(['cancer', 'walk_distance', 'walk_amount']).agg({'distance': 'mean'}).reset_index()

    # Create a unique identifier for each combination
    group['combo'] = group.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Create a DataFrame with all possible combinations
    all_combos_df = pd.DataFrame(all_combos, columns=['walk_distance', 'walk_amount'])
    all_combos_df['combo'] = all_combos_df.apply(lambda row: f"{row['walk_distance']}_{row['walk_amount']}", axis=1)

    # Merge to ensure all combinations are present, filling missing with NaN
    merged = pd.merge(all_combos_df, group, on=['walk_distance', 'walk_amount', 'combo'], how='left')

    # Sort by walk_distance and walk_amount
    merged = merged.sort_values(['walk_distance', 'walk_amount'])

    # Assign each combination a unique angle
    N = len(all_combos)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Ensure the plot is circular by appending the first value at the end
    angles += angles[:1]

    # Create a mapping from combo to angle
    combo_to_angle = dict(zip(all_combos_df['combo'], angles[:-1]))

    # Set the labels for each combination
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_combos_df['combo'], fontsize=10)

    # Iterate over each cancer type and plot its inter-class distances
    cancers = group['cancer'].unique()
    for cancer in cancers:
        cancer_data = merged[merged['cancer'] == cancer].sort_values(['walk_distance', 'walk_amount'])
        combos = cancer_data['combo'].tolist()
        distances = cancer_data['distance'].tolist()

        # Handle missing distances by setting them to 0
        distances = [d if not np.isnan(d) else 0 for d in distances]

        # Append the first distance to close the loop
        distances += distances[:1]
        cancer_angles = [combo_to_angle[combo] for combo in combos] + [combo_to_angle[combos[0]]]

        # Plot the line
        ax.plot(cancer_angles, distances, label=cancer, color=color_dict.get(cancer, None), linewidth=2)

        # Plot the points
        ax.scatter(cancer_angles, distances, color=color_dict.get(cancer, None), s=50, edgecolors='w', zorder=5)

    # Set the title
    ax.set_title("Inter-Class Distances", va='bottom')

    # Optional: Set radial limits with some padding
    max_distance = merged['distance'].max()
    ax.set_ylim(0, max_distance * 1.1)


def main_polar_plots(combined_df: pd.DataFrame, file_name: str):
    """
    Generates two separate polar plots for intra and inter distances with enhanced legends.
    """
    # Prepare color mapping for cancers
    cancers = combined_df['cancer'].unique()
    num_cancers = len(cancers)

    print(f"Number of cancers: {num_cancers}")

    # Use a color palette with enough distinct colors
    palette = sns.color_palette("hsv", num_cancers)
    color_dict = dict(zip(cancers, palette))

    # Define all possible combinations
    all_combos = [(wd, wa) for wd in walk_distances for wa in walk_amounts]

    # Create subplots: one for intra and one for inter
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    # Create intra distances polar plot
    create_polar_line_plot(combined_df, 'intra', axes[0], color_dict, all_combos)

    # Create inter distances polar plot
    create_polar_inter_plot(combined_df, axes[1], color_dict, all_combos)

    # Create a single legend for both plots
    # Extract handles and labels from the first plot
    handles, labels = axes[0].get_legend_handles_labels()

    # Place the legend outside the plots
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, title="Cancer Types")

    # Add a main title
    plt.suptitle("Polar Plots of Intra and Inter Cancer Type Distances", fontsize=18, fontweight='bold', y=1.02)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(Path(figure_save_folder, f"{file_name}.png"), bbox_inches='tight', dpi=300)


def calculate_intra_inter_distances(summed_embeddings: dict, selected_cancers: list,
                                    distance_metric: str = "euclidean"):
    intra_distances = {}
    inter_distances = {}

    # Process each (walk_distance, walk_amount) combination separately
    for (walk_distance, walk_amount), df in summed_embeddings.items():
        cancer_dfs = {}

        # Iterate over each selected cancer type
        for cancer in selected_cancers:
            # Select rows where the specific cancer type is present
            cancer_rows = df[df["cancer"] == cancer].copy()

            if cancer_rows.empty:
                print(
                    f"Warning: No data for cancer type '{cancer}' with walk_distance={walk_distance} and walk_amount={walk_amount}.")
                continue

            # Drop unwanted columns
            cancer_rows = cancer_rows.drop(columns=["cancer"])

            # Ensure that unwanted columns are indeed dropped
            assert "cancer" not in cancer_rows.columns, "Cancer column is present"

            # Store the filtered DataFrame
            cancer_dfs[cancer] = cancer_rows

        # Calculate intra-class (within the same cancer type) distances
        for cancer, df_cancer in cancer_dfs.items():
            if df_cancer.empty:
                print(
                    f"Warning: No data for cancer type '{cancer}' with walk_distance={walk_distance} and walk_amount={walk_amount}.")
                continue

            if distance_metric == "euclidean":
                intra_distance = euclidean_distances(df_cancer)
            elif distance_metric == "cosine":
                intra_distance = cosine_distances(df_cancer)
            elif distance_metric == "dot_product":
                intra_distance = dot_product_distance(df_cancer)
            else:
                raise ValueError(f"Invalid distance metric: {distance_metric}")

            # Extract the upper triangle without the diagonal
            intra_distance = intra_distance[np.triu_indices_from(intra_distance, k=1)]
            intra_key = (walk_distance, walk_amount, cancer)
            intra_distances[intra_key] = intra_distance

        # Calculate inter-class (between different cancer types) distances
        cancers_list = list(cancer_dfs.keys())
        for i in range(len(cancers_list)):
            for j in range(i + 1, len(cancers_list)):
                cancer1 = cancers_list[i]
                cancer2 = cancers_list[j]
                df1 = cancer_dfs[cancer1]
                df2 = cancer_dfs[cancer2]

                if df1.empty or df2.empty:
                    print(f"Warning: One of the DataFrames for cancers '{cancer1}' or '{cancer2}' is empty.")
                    continue

                if distance_metric == "euclidean":
                    inter_distance = euclidean_distances(df1, df2).flatten()
                elif distance_metric == "cosine":
                    inter_distance = cosine_distances(df1, df2).flatten()
                elif distance_metric == "dot_product":
                    inter_distance = dot_product_distance(df1, df2).flatten()
                else:
                    raise ValueError(f"Invalid distance metric: {distance_metric}")

                inter_key = (walk_distance, walk_amount, cancer1, cancer2)
                inter_distances[inter_key] = inter_distance

    return intra_distances, inter_distances


def convert_to_records(intra_df: {}, inter_df: {}):
    # Convert intra_distances to a DataFrame
    intra_records = []
    for key, distances in intra_df.items():
        walk_distance, walk_amount, cancer = key
        for distance in distances:
            intra_records.append({
                "type": "intra",
                "walk_distance": walk_distance,
                "walk_amount": walk_amount,
                "cancer": cancer,
                "distance": distance,
                "combined_cancer": cancer
            })
    intra_df = pd.DataFrame(intra_records)

    # Convert inter_distances to a DataFrame
    inter_records = []
    for key, distances in inter_df.items():
        walk_distance, walk_amount, cancer1, cancer2 = key
        for distance in distances:
            inter_records.append({
                "type": "inter",
                "walk_distance": walk_distance,
                "walk_amount": walk_amount,
                "cancer1": cancer1,
                "cancer2": cancer2,
                "distance": distance,
                "combined_cancer": f"{cancer1}-{cancer2}"
            })
    inter_df = pd.DataFrame(inter_records)

    return intra_df, inter_df


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=False, help="The cancer type to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
                        )
    args = parser.parse_args()
    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    figure_save_folder = Path(figure_save_folder, cancers, "distances")
    results_save_folder = Path(results_save_folder, cancers)

    if not figure_save_folder.exists():
        figure_save_folder.mkdir(parents=True)

    if not results_save_folder.exists():
        results_save_folder.mkdir(parents=True)

    summed_embeddings = {}

    # Load and concatenate embeddings for each (walk_distance, walk_amount) combination
    for walk_distance in walk_distances:
        for walk_amount in walk_amounts:
            cancers = "_".join(selected_cancers)
            csv_path = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}",
                            "summed_embeddings.h5")

            if not csv_path.exists():
                print(f"Warning: {csv_path} does not exist. Skipping.")
                continue

            # load hfd5 file
            with h5py.File(csv_path, 'r') as h5_file:
                df = pd.DataFrame(h5_file['X'][:])
                df['cancer'] = h5_file['y'][:].astype(str)
                df['walk_distance'] = walk_distance
                df['walk_amount'] = walk_amount

                # remove all na rows where cancer is NAN
                df = df.dropna(subset=['cancer'])

            # Ensure 'cancer' columns are present before processing
            if "cancer" not in df.columns:
                print(f"Warning: 'cancer' column is missing in {csv_path}. Skipping this file.")
                continue

            key = (walk_distance, walk_amount)
            if key not in summed_embeddings:
                summed_embeddings[key] = [df]
            else:
                summed_embeddings[key].append(df)

    # Concatenate all DataFrames for each (walk_distance, walk_amount) key
    for key, dfs in summed_embeddings.items():
        summed_embeddings[key] = pd.concat(dfs, ignore_index=True)

    # Initialize dictionaries to store distances
    euclidean_intra_distances, euclidean_inter_distances = calculate_intra_inter_distances(summed_embeddings,
                                                                                           selected_cancers,
                                                                                           distance_metric="euclidean")
    cosine_intra_distances, cosine_inter_distances = calculate_intra_inter_distances(summed_embeddings,
                                                                                     selected_cancers,
                                                                                     distance_metric="cosine")
    dot_product_intra_distances, dot_product_inter_distances = calculate_intra_inter_distances(summed_embeddings,
                                                                                               selected_cancers,
                                                                                               distance_metric="dot_product")

    # Convert to DataFrames
    euclidean_intra_df, euclidean_inter_df = convert_to_records(euclidean_intra_distances, euclidean_inter_distances)
    cosine_intra_df, cosine_inter_df = convert_to_records(cosine_intra_distances, cosine_inter_distances)
    dot_product_intra_df, dot_product_inter_df = convert_to_records(dot_product_intra_distances,
                                                                    dot_product_inter_distances)

    # Combine intra and inter distance DataFrames
    euclidean_combined_df = pd.concat([euclidean_intra_df, euclidean_inter_df], ignore_index=True)
    cosine_combined_df = pd.concat([cosine_intra_df, cosine_inter_df], ignore_index=True)
    dot_product_combined_df = pd.concat([dot_product_intra_df, dot_product_inter_df], ignore_index=True)

    euclidean_save_file_name: str = "euclidean_combined_distances.csv"
    cosine_save_file_name: str = "cosine_combined_distances.csv"
    dot_product_save_file_name: str = "dot_product_combined_distances.csv"

    # Save the combined DataFrame
    euclidean_combined_df.to_csv(Path(results_save_folder, euclidean_save_file_name), index=False)
    cosine_combined_df.to_csv(Path(results_save_folder, cosine_save_file_name), index=False)
    dot_product_combined_df.to_csv(Path(results_save_folder, dot_product_save_file_name), index=False)

    # Generate the polar plots with enhanced legends
    main_polar_plots(euclidean_combined_df, "euclidean_polar")
    main_polar_plots(cosine_combined_df, "cosine_polar")
    main_polar_plots(dot_product_combined_df, "dot_product_polar")
