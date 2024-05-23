import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_folder = Path("results", "embeddings")
save_folder = Path("results", "summed_embeddings", "simple_embeddings")


def load_embeddings(load_folder):
    # Load the embeddings from CSV files
    rna_embeddings = pd.read_csv(Path(load_folder, "embeddings", "rna_embeddings.csv"))
    sentence_embeddings = pd.read_csv(Path(load_folder, "embeddings", "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "embeddings", "image_embeddings.csv"))
    return rna_embeddings, sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, max_count):
    # Randomly choose embeddings up to max_count
    n = random.randint(1, max_count)  # Ensure at least one is selected
    chosen_indices = random.sample(range(len(embeddings)), n)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, n


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--embeddings", "-e", type=int, help="Number of embeddings to sum")
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    parser.add_argument("--load_folder", "-l", type=str, required=True, help="Folder containing the embeddings")
    args = parser.parse_args()

    iterations = args.iterations
    total_embeddings = args.embeddings
    load_folder = Path(args.load_folder)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load embeddings
    rna_embeddings, sentence_embeddings, image_embeddings = load_embeddings(args.load_folder)

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Determine random order for processing embeddings
        embeddings_list = [(rna_embeddings, 'RNA'), (sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        random.shuffle(embeddings_list)

        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = total_embeddings
        combination_counts = {}

        for embeddings, name in embeddings_list:
            if remaining_embeddings > 0:
                current_sum, count = random_sum_embeddings(embeddings, remaining_embeddings)
                combined_sum += current_sum
                remaining_embeddings -= count
                combination_counts[name] = count
            else:
                combination_counts[name] = 0

        # Ensure there is at least one embedding selected in total (avoid all-zero entries)
        if all(v == 0 for v in combination_counts.values()):
            embeddings, name = random.choice(embeddings_list)
            current_sum, count = random_sum_embeddings(embeddings, 1)  # Force at least one selection
            combined_sum += current_sum
            combination_counts[name] = count

        # Combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + [combination_counts['Text'], combination_counts['Image'],
                                                   combination_counts['RNA']])

    # Save the data to CSV
    column_names = list(embeddings_list[0][0].columns) + ["Text", "Image", "RNA"]
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    # convert all columns to int
    combined_df = combined_df.astype(float)
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{total_embeddings}_embeddings.csv"), index=False)
