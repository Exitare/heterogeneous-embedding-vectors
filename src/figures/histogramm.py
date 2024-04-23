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
    plt.xticks(np.unique(y_test))
    plt.tight_layout()
    plt.savefig(Path("figures", "recognizer", "histogram.png"), dpi=300)
