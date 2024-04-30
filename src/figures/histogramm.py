import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

embeddings = ['Text', 'Image', 'RNA']
load_folder = Path("results", "recognizer")

if __name__ == '__main__':
    # load y_test data
    y_test = np.load(Path(load_folder, "y_test.npy"))

    # plot histogram for each output in one plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, y in enumerate(y_test.T):
        axs[i].hist(y, bins=4, color='blue', edgecolor='black', alpha=0.7)
        axs[i].set_title(f"Embedding Sum Distribution for {embeddings[i]}")
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Frequency")
        # change x axis to only the distinct values of the y_test
        axs[i].set_xticks(np.unique(y))
    plt.tight_layout()
    plt.savefig(Path("figures", "recognizer", "histogram.png"), dpi=300)
    plt.close('all')


    # count unique combination of Image, Text and RNA embeddings
    # count the unique combinations of the embeddings
    unique_combinations = pd.DataFrame(y_test).drop_duplicates()
    unique_combinations = unique_combinations.groupby(list(range(3))).size().reset_index(name='count')
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(unique_combinations[0], unique_combinations[1], unique_combinations[2], c='blue', marker='o')

    # Labels
    ax.set_xlabel('Column 0')
    ax.set_ylabel('Column 1')
    ax.set_zlabel('Column 2')

    # Title
    ax.set_title('3D Scatter Plot of Unique Combinations')

    plt.savefig(Path("figures", "recognizer", "scatter.png"), dpi=300)
    plt.close('all')
