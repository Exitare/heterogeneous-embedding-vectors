from argparse import ArgumentParser
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

modalities = ["RNA", "Mutation", "Image", "Text"]

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("-n", "--noise", help="The amount of noise to add", type=float, required=False, default=0.0)
    parser.add_argument("-w", "--walk_distance", help="The distance of the walk", type=int, required=False, default=3)

    args = parser.parse_args()

    selected_cancers: [str] = args.cancer
    cancers: str = "_".join(selected_cancers)
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    noise: float = args.noise
    walk_distance: int = args.walk_distance

    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Amount of walk embeddings: {amount_of_walk_embeddings}")
    logging.info(f"Noise: {noise}")
    logging.info(f"Walk distance: {walk_distance}")

    h5_file_path = Path(
        f"results/recognizer/summed_embeddings/multi/{cancers}/{amount_of_walk_embeddings}/{noise}/{walk_distance}_embeddings.h5")

    # Open the HDF5 file
    with h5py.File(h5_file_path, "r") as h5_file:

        # Iterate through keys in the HDF5 file, excluding 'X'
        for key in h5_file.keys():
            if key == 'X':
                continue

            data = h5_file[key][:]  # Load the dataset

            # Determine whether to use a pie chart or histogram
            if np.issubdtype(data.dtype, np.integer) and len(np.unique(data)) <= 10:
                # Use a pie chart for categorical-like integer distributions with <10 unique values
                unique, counts = np.unique(data, return_counts=True)

                plt.figure(figsize=(8, 8))
                wedges, _, autotexts = plt.pie(
                    counts,
                    labels=None,  # Remove labels
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=1.2,  # Move percentages outside the pie
                    textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'}  # Improve readability
                )

                # Rotate the percentage labels by 90 degrees
                for autotext in autotexts:
                    autotext.set_rotation(90)

                # Use a legend instead of direct labels
                plt.legend(wedges, unique, title=key, loc="center left", bbox_to_anchor=(1, 0.5))
                plt.title(f"Distribution of {key}")

            else:
                # Use a histogram for continuous or high-cardinality data
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
                plt.xlabel(key)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {key}")

            # ===== NEW PIE CHART: Zero vs. Non-Zero Distribution =====
            zero_count = np.sum(data == 0)
            non_zero_count = np.sum(data != 0)

            plt.figure(figsize=(8, 8))
            wedges, _, autotexts = plt.pie(
                [zero_count, non_zero_count],
                labels=["Zero", "Non-Zero"],
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=1.2,
                textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'}
            )

            # Rotate percentage labels
            for autotext in autotexts:
                autotext.set_rotation(90)

            plt.title(f"Zero vs. Non-Zero Distribution in {key}")
            plt.show()
