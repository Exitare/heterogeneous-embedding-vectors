import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

save_folder = Path("results", "noise_recognizer", "summed_embeddings", "simple")
load_folder = Path("results", "embeddings")


def load_embeddings():
    # Load the embeddings from CSV files
    rna_embeddings = pd.read_csv(Path(load_folder, "rna_embeddings.csv"))
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return rna_embeddings, sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, count, add_noise=False, scramble=False):
    if add_noise:
        # Add random noise vectors of the same shape as embeddings
        noise_vectors = pd.DataFrame(np.random.uniform(-1, 1, size=(count, embeddings.shape[1])),
                                     columns=embeddings.columns)
        chosen_embeddings = noise_vectors  # Use noise instead of actual embeddings
    else:
        # Randomly choose the specified number of embeddings
        chosen_indices = random.sample(range(len(embeddings)), count)
        chosen_embeddings = embeddings.iloc[chosen_indices]

    if scramble:
        # Scramble the vectors within the embeddings
        chosen_embeddings = chosen_embeddings.apply(np.random.permutation)

    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, count


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum")
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0, help="Ratio of random noise vectors to add")
    parser.add_argument("--scramble", "-s", action="store_true", help="Whether to scramble the embeddings")
    args = parser.parse_args()

    iterations = args.iterations
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio
    scramble = args.scramble

    save_folder = Path(save_folder, str(iterations))

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load embeddings
    rna_embeddings, sentence_embeddings, image_embeddings = load_embeddings()

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Determine random order for processing embeddings
        embeddings_list = [(rna_embeddings, 'RNA'), (sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        random.shuffle(embeddings_list)

        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = walk_distance
        combination_counts = {}

        for i, (embeddings, name) in enumerate(embeddings_list):
            # Calculate the maximum number of embeddings that can be selected
            max_embeddings_for_type = remaining_embeddings - (len(embeddings_list) - i - 1)

            if i < len(embeddings_list) - 1:  # Not the last type
                count = random.randint(1, max(max_embeddings_for_type, 1))
            else:  # Last type must take all remaining embeddings
                count = remaining_embeddings

            # Decide if we should add noise
            add_noise = random.random() < noise_ratio
            current_sum, count = random_sum_embeddings(embeddings, count, add_noise=add_noise, scramble=scramble)
            combined_sum += current_sum
            remaining_embeddings -= count
            combination_counts[name] = count

        # Ensure the total number of selected embeddings equals walk_distance
        total_selected = sum(combination_counts.values())
        assert total_selected == walk_distance, f"Total embeddings selected ({total_selected}) does not match walk_distance ({walk_distance})"

        # Combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + [combination_counts['Text'], combination_counts['Image'],
                                                   combination_counts['RNA']])

    # Save the data to CSV
    column_names = list(embeddings_list[0][0].columns) + ["Text", "Image", "RNA"]
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    # Convert all columns to float
    combined_df = combined_df.astype(float)

    # Print a message and save the combined embeddings to CSV
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
