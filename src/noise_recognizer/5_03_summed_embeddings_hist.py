from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # python3 src/recognizer/multi_cancer_recognizer.py -e 5 -c blca brca
    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--walk_distance', "-w", type=int, required=True,
                        help='The number of the walk distance to work with.')
    parser.add_argument("--cancer", "-c", nargs="+", required=True,
                        help="The cancer types to work with, e.g. blca brca")
    args = parser.parse_args()

    walk_distance = args.walk_distance
    cancers = args.cancer

    selected_cancers = "_".join(cancers)

    summed_embeddings = pd.read_csv(f"results/noise_recognizer/summed_embeddings/multi/{selected_cancers}/{walk_distance}_embeddings.csv")

    print(summed_embeddings)

    # Plot histograms for RNA, Text, and Image columns
    columns_to_plot = ['RNA', 'Text', 'Image']

    # Create subplots
    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(15, 5))

    for i, column in enumerate(columns_to_plot):
        axes[i].hist(summed_embeddings[column], bins=walk_distance, edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        # set the x-axis to a range of 0 to walk_distance
        axes[i].set_xlim(0, walk_distance)
        # show all integer values on the x-axis
        axes[i].set_xticks(range(walk_distance + 1))

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()