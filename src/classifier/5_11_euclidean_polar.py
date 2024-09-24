from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Corrected walk_distances to include 4
walk_distances = [3, 4, 5]
walk_amounts = [3, 4, 5]


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
    ax.set_title("Inter-Class Distances", va='bottom', fontsize=14, fontweight='bold')

    # Optional: Set radial limits with some padding
    max_distance = merged['distance'].max()
    ax.set_ylim(0, max_distance * 1.1)


def main_polar_plots(combined_df):
    """
    Generates two separate polar plots for intra and inter distances with enhanced legends.
    """
    # Prepare color mapping for cancers
    cancers = combined_df['cancer'].unique()
    num_cancers = len(cancers)

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
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, title="Cancer Types",
               fontsize=12)

    # Add a main title
    plt.suptitle("Polar Plots of Intra and Inter Cancer Type Distances", fontsize=18, fontweight='bold', y=1.02)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig("polar_distance_lineplots_with_legend.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="The cancer type to work with.")
    args = parser.parse_args()
    selected_cancers = args.cancer

    summed_embeddings = {}

    # Load and concatenate embeddings for each (walk_distance, walk_amount) combination
    for walk_distance in walk_distances:
        for walk_amount in walk_amounts:
            cancers = "_".join(selected_cancers)
            csv_path = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}",
                            "summed_embeddings.csv")

            if not csv_path.exists():
                print(f"Warning: {csv_path} does not exist. Skipping.")
                continue

            df = pd.read_csv(csv_path)
            df["walk_distance"] = walk_distance
            df["walk_amount"] = walk_amount

            # Ensure 'patient_id' and 'cancer' columns are present before processing
            if "patient_id" not in df.columns:
                print(f"Warning: 'patient_id' column is missing in {csv_path}. Skipping this file.")
                continue
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
            cancer_rows = cancer_rows.drop(columns=["patient_id", "cancer"])

            # Ensure that unwanted columns are indeed dropped
            assert "patient_id" not in cancer_rows.columns, "Patient id column is present"
            assert "cancer" not in cancer_rows.columns, "Cancer column is present"

            # Store the filtered DataFrame
            cancer_dfs[cancer] = cancer_rows

        # Calculate intra-class (within the same cancer type) distances
        for cancer, df_cancer in cancer_dfs.items():
            if df_cancer.empty:
                print(
                    f"Warning: No data for cancer type '{cancer}' with walk_distance={walk_distance} and walk_amount={walk_amount}.")
                continue

            intra_distance = euclidean_distances(df_cancer)
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

                inter_distance = euclidean_distances(df1, df2).flatten()
                inter_key = (walk_distance, walk_amount, cancer1, cancer2)
                inter_distances[inter_key] = inter_distance

    # Convert intra_distances to a DataFrame
    intra_records = []
    for key, distances in intra_distances.items():
        walk_distance, walk_amount, cancer = key
        for distance in distances:
            intra_records.append({
                "type": "intra",
                "walk_distance": walk_distance,
                "walk_amount": walk_amount,
                "cancer": cancer,
                "distance": distance
            })
    intra_df = pd.DataFrame(intra_records)

    # Convert inter_distances to a DataFrame
    inter_records = []
    for key, distances in inter_distances.items():
        walk_distance, walk_amount, cancer1, cancer2 = key
        for distance in distances:
            inter_records.append({
                "type": "inter",
                "walk_distance": walk_distance,
                "walk_amount": walk_amount,
                "cancer1": cancer1,
                "cancer2": cancer2,
                "distance": distance
            })
    inter_df = pd.DataFrame(inter_records)

    # Combine intra and inter distance DataFrames
    combined_df = pd.concat([intra_df, inter_df], ignore_index=True)

    # Save the combined DataFrame
    combined_df.to_csv("combined_distances.csv", index=False)
    print("Combined distance DataFrame has been saved to 'combined_distances.csv'.")

    # Verify that all 6 cancer types are present
    print("Cancer Types Present:", combined_df['cancer'].unique())

    # Generate the polar plots with enhanced legends
    main_polar_plots(combined_df)
