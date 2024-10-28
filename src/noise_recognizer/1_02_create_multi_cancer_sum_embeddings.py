import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_folder = Path("results", "embeddings")
cancer_embedding_load_folder = Path(load_folder, "cancer")
save_folder = Path("results", "noise_recognizer", "summed_embeddings", "multi")


def load_noise_embeddings():
    # Load the embeddings from CSV files
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, count, add_noise=False):
    if add_noise:
        # Add random noise vectors of the same shape as embeddings
        noise_vectors = pd.DataFrame(np.random.uniform(-1, 1, size=(count, embeddings.shape[1])),
                                     columns=embeddings.columns)
        chosen_embeddings = noise_vectors
    else:
        # Randomly choose the specified number of embeddings
        chosen_indices = random.sample(range(len(embeddings)), count)
        chosen_embeddings = embeddings.iloc[chosen_indices]

    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, count


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True,
                        help="The selected cancer identifier to sum")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1,
                        help="Ratio of noise vectors to add (0.0 - 1.0)")
    args = parser.parse_args()

    iterations = args.iterations
    walk_distance = args.walk_distance
    selected_cancers = args.selected_cancers
    noise_ratio = args.noise_ratio

    cancers = "_".join(selected_cancers)

    cancer_embedding_load_folder = Path(cancer_embedding_load_folder, cancers)
    print("Loading embeddings from:", cancer_embedding_load_folder)
    save_folder = Path(save_folder, cancers)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    print(f"Using save folder: {save_folder}")

    # Dynamically load all embeddings
    loaded_cancer_embeddings = {}
    for cancer in selected_cancers:
        try:
            temp_df = pd.read_csv(Path(cancer_embedding_load_folder, f"{cancer.lower()}_embeddings.csv"))
            # remove patient column if exists
            if "Patient" in temp_df.columns:
                temp_df.drop(columns=["Patient"], inplace=True)
            loaded_cancer_embeddings[cancer] = temp_df
        except FileNotFoundError:
            print(f"Could not load {cancer} embedding. Skipping this cancer type.")
            continue

    if not loaded_cancer_embeddings:
        raise ValueError("No valid cancer embeddings were loaded. Please check your input.")

    # Load noise embeddings
    sentence_embeddings, image_embeddings = load_noise_embeddings()

    # Prepare a list of all available modalities dynamically
    modality_names = ['Text', 'Image'] + list(loaded_cancer_embeddings.keys())
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Create dynamic list of embeddings with modality names
        embeddings_list = [(sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        for cancer_type, cancer_embedding in loaded_cancer_embeddings.items():
            embeddings_list.append((cancer_embedding, cancer_type))

        random.shuffle(embeddings_list)

        # Initialize combined sum and remaining embeddings counter
        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = walk_distance
        combination_counts = {name: 0 for name in modality_names}

        # Process embeddings and apply noise or scrambling
        for i, (embeddings, name) in enumerate(embeddings_list):
            if remaining_embeddings == 0:
                break

            # Calculate how many embeddings to use for this modality
            if i == len(embeddings_list) - 1:
                # Ensure the last type gets all remaining embeddings
                max_embeddings_for_type = remaining_embeddings
            else:
                # Randomly choose how many embeddings to use
                max_embeddings_for_type = random.randint(0, remaining_embeddings)

            # Determine if we should add noise based on noise_ratio
            add_noise = random.random() < noise_ratio

            # Select random sum embedding with noise or scrambling
            current_sum, count = random_sum_embeddings(embeddings, max_embeddings_for_type, add_noise=add_noise)

            # Update the combined sum
            combined_sum += current_sum

            # Update the remaining embeddings
            remaining_embeddings -= count

            # Update the combination counts
            if name in combination_counts:
                combination_counts[name] += count
            else:
                combination_counts[name] = count

            if remaining_embeddings < 0:
                raise ValueError(
                    "Selected more embeddings than remaining. Check the random_sum_embeddings function or logic.")

        # Validate that total embeddings selected match walk distance
        if sum(combination_counts.values()) != walk_distance:
            raise ValueError("Total number of embeddings selected does not match walk distance!")

        # Append combined sum and counts for each modality dynamically
        combined_row = list(combined_sum) + [combination_counts[modality] for modality in modality_names]
        combined_data.append(combined_row)

    # Define column names dynamically for CSV export
    column_names = list(embeddings_list[0][0].columns) + modality_names

    combined_df = pd.DataFrame(combined_data, columns=column_names)

    # Add RNA column as the sum of all cancer types in the dataset
    combined_df["RNA"] = combined_df[list(loaded_cancer_embeddings.keys())].sum(axis=1)

    # Convert all columns to float for consistency
    combined_df = combined_df.astype(float)

    # Convert Text, Image, and RNA columns to integers
    combined_df["Text"] = combined_df["Text"].astype(int)
    combined_df["Image"] = combined_df["Image"].astype(int)
    combined_df["RNA"] = combined_df["RNA"].astype(int)

    for cancer in loaded_cancer_embeddings.keys():
        combined_df[cancer] = combined_df[cancer].astype(int)

    print(combined_df)

    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
    print("Done!")
