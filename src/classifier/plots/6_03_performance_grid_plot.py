from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

save_folder = Path("figures", "classifier")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, "performance")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    results = []
    # iterate over all subfolders
    cancer_folder = Path("results", "classifier", "classification", cancers)
    for run in cancer_folder.iterdir():
        if run.is_file():
            continue

        # iterate over all subfolders
        for sub_run in run.iterdir():
            if sub_run.is_file():
                continue
            try:
                df = pd.read_csv(Path(sub_run, "results.csv"))
                # load the results from the run
                results.append(df)
            except FileNotFoundError:
                continue

    # concatenate all results
    results = pd.concat(results)
    print(results)
    print(results["walk_distance"].unique())
    print(results["amount_of_walks"].unique())

    pivot = results.pivot_table(
        values='accuracy',
        index=['cancer'],
        columns=['walk_distance', 'amount_of_walks'],
        aggfunc='first'
    )

    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define color palette
    colors = sns.color_palette("husl", n_colors=len(pivot.columns))

    # Plot grouped bars
    bar_width = 0.15
    for i, col in enumerate(pivot.columns):
        x = np.arange(len(pivot.index))
        offset = (i - len(pivot.columns) / 2 + 0.5) * bar_width
        ax.bar(x + offset, pivot[col], width=bar_width, label=f'WD:{col[0]}, AW:{col[1]}', color=colors[i])

    # Customize the plot
    ax.set_xlabel('Cancer Types')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cancer Classification Accuracy Across Walk Distances and Amount of Walks')
    ax.set_xticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(title='Walk Distance (WD) : Amount of Walks (AW)', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add a grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display the plot
    plt.tight_layout()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Plot for Walk Distance
    sns.lineplot(data=results, x='walk_distance', y='accuracy', hue='cancer', marker='o', ax=ax1)
    ax1.set_title('Accuracy vs Walk Distance')
    ax1.set_xlabel('Walk Distance')
    ax1.set_ylabel('Accuracy')
    ax1.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot for Amount of Walks
    sns.lineplot(data=results, x='amount_of_walks', y='accuracy', hue='cancer', marker='o', ax=ax2)
    ax2.set_title('Accuracy vs Amount of Walks')
    ax2.set_xlabel('Amount of Walks')
    ax2.set_ylabel('Accuracy')
    ax2.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"grid_performance.png"), dpi=150)
    plt.close('all')

    # Create a new column to differentiate the type of plot
    results_walk_distance = results.copy()
    results_walk_distance['type'] = 'Walk Distance'
    results_walk_distance.rename(columns={'walk_distance': 'x_value'}, inplace=True)

    results_amount_of_walks = results.copy()
    results_amount_of_walks['type'] = 'Amount of Walks'
    results_amount_of_walks.rename(columns={'amount_of_walks': 'x_value'}, inplace=True)

    # Set up the FacetGrid
    g = sns.FacetGrid(results, col="walk_distance", row="amount_of_walks", hue="cancer", margin_titles=True,
                      palette="tab10")

    # Map the scatter plot to the grid
    g.map(sns.barplot, "cancer", "accuracy", alpha=.7)

    # Rotate x-axis labels for each subplot
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Add legends and adjust layout
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    # rename walk_distance to Walk Distance in facet grid
    g.set_axis_labels("Cancer Type", "Accuracy")
    # rename amount_of_walks to Amount of Walks in facet grid
    g.set_titles(col_template="Walk Distance: {col_name}", row_template="Amount of Walks: {row_name}")

    g.fig.suptitle('Model Accuracy Across Different Walk Distances and Amount of Walks')

    plt.savefig(Path(save_folder, f"facet_performance.png"), dpi=300)

    # Combine the two DataFrames
    combined_results = pd.concat([results_walk_distance[['x_value', 'accuracy', 'cancer', 'type']],
                                  results_amount_of_walks[['x_value', 'accuracy', 'cancer', 'type']]])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_results, x='x_value', y='accuracy', hue='cancer', style='type',
                 palette={"BRCA": "red", "BLCA": "blue", "LUAD": "green", "STAD": "purple", "THCA": "orange",
                          "COAD": "yellow" , "All": "pink"}, markers=True)

    plt.title('Accuracy vs Walk Distance and Amount of Walks')
    plt.xlabel('Value')
    plt.ylabel('Accuracy')
    plt.legend(title='Cancer Type and Walk Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    # set x axis ticks to 3,4 and 5
    plt.xticks([3, 4, 5])
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"combined_performance.png"), dpi=300)
