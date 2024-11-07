import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

save_folder = Path("results", "noise_recognizer", "summed_embeddings", "simple")
load_folder = Path("results", "recognizer_embeddings")


def load_embeddings():
    # Load the embeddings from CSV files
    rna_embeddings = pd.read_csv(Path(load_folder, "rna_embeddings.csv"))
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return rna_embeddings, sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, count, add_noise=False):
    if add_noise:
        # Add random noise vectors of the same shape as embeddings
        noise_vectors = pd.DataFrame(np.random.uniform(-1, 1, size=(count, embeddings.shape[1])),
                                     columns=embeddings.columns)
        chosen_embeddings = noise_vectors  # Use noise instead of actual embeddings
        is_noise_only = True
    else:
        # Randomly choose the specified number of embeddings
        chosen_indices = random.sample(range(len(embeddings)), count)
        chosen_embeddings = embeddings.iloc[chosen_indices]
        is_noise_only = False

    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, count, is_noise_only


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum")
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1, help="Ratio of random noise vectors to add")
    args = parser.parse_args()

    iterations = args.iterations
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio

    save_folder = Path(save_folder, str(iterations), str(noise_ratio))

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load embeddings
    rna_embeddings, sentence_embeddings, image_embeddings = load_embeddings()

    # drop submitter_id	cancer_type	tile_pos columns
    image_embeddings.drop(columns=["submitter_id", "cancer_type", "tile_pos"], inplace=True)

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(iterations)):
        embeddings_list = [(rna_embeddings, 'RNA'), (sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        random.shuffle(embeddings_list)

        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = walk_distance
        combination_counts = {'Text': 0, 'Image': 0, 'RNA': 0}
        total_noise = True  # Flag to check if only noise was added

        for i, (embeddings, name) in enumerate(embeddings_list):
            max_embeddings_for_type = remaining_embeddings - (len(embeddings_list) - i - 1)

            if i < len(embeddings_list) - 1:
                count = random.randint(1, max(max_embeddings_for_type, 1))
            else:
                count = remaining_embeddings

            # Decide if we should add noise
            add_noise = random.random() < noise_ratio
            current_sum, count, is_noise_only = random_sum_embeddings(embeddings, count, add_noise=add_noise)

            combined_sum += current_sum
            remaining_embeddings -= count

            if not is_noise_only:
                combination_counts[name] = count  # Update only if not noise
                total_noise = False  # At least one real embedding was used

        # If total noise was added, set all combination counts to 0
        if total_noise:
            combination_counts = {'Text': 0, 'Image': 0, 'RNA': 0}

        # Combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + [combination_counts['Text'], combination_counts['Image'],
                                                   combination_counts['RNA']])

    # Define column names
    column_names = list(embeddings_list[0][0].columns) + ["Text", "Image", "RNA"]

    # Create DataFrame
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    combined_df = combined_df.astype(float)  # Convert all columns to float

    # Print a message and save the combined embeddings to CSV
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
