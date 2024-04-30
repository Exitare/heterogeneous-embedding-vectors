import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# load folder
load_folder = Path("results", "recognizer")
save_folder = Path("figures", "recognizer")

# Plot confusion matrices
embeddings = ['Text', 'Image', 'RNA']

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)
    # Load data using numpy
    y_test_binary = np.load(Path(load_folder, "y_test_binary.npy"))
    y_pred_binary = np.load(Path(load_folder, "y_pred_binary.npy"))

    y_test = np.load(Path(load_folder, "y_test.npy"))
    y_pred = np.load(Path(load_folder, "y_pred.npy"))
    y_pred = [np.squeeze(pred) for pred in y_pred]  # Remove unnecessary dimensions

    binary_conf_matrices = [confusion_matrix(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]
    conf_matrices = [confusion_matrix(y_test[:, i], y_pred[i]) for i in range(y_test.shape[1])]

    # Print confusion matrices
    for i, cm in enumerate(conf_matrices):
        print(f"Confusion Matrix for output {i} (Text, Image, RNA - respective order):")
        print(cm)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(cm, annot=True, fmt='d', ax=axs[i])
        axs[i].set_title(f"Confusion matrix for {embeddings[i]}")
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(Path(save_folder, "confusion_matrices.png"), dpi=300)

    # plot binary confusion matrices
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, cm in enumerate(binary_conf_matrices):
        sns.heatmap(cm, annot=True, fmt='d', ax=axs[i])
        axs[i].set_title(f"Binary confusion matrix for {embeddings[i]}")
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(Path(save_folder, "confusion_matrices_binary.png"), dpi=300)
