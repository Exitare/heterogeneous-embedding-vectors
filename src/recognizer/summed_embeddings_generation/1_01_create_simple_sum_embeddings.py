import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

save_folder = Path("results", "recognizer", "summed_embeddings", "simple")
load_folder = Path("results", "embeddings")


def load_rna_embeddings(cancers: []):
    rna_embeddings = []
    cancer_load_folder = Path(load_folder, "rna", cancers)
    for file in Path(cancer_load_folder).iterdir():
        if file.is_file() and file.name.endswith("_embeddings.csv"):
            rna_embeddings.append(pd.read_csv(file))

    rna_embeddings = pd.concat(rna_embeddings, ignore_index=True)
    return rna_embeddings


def load_embeddings(cancers: []):
    # Load the embeddings from CSV files
    sentence_embeddings = pd.read_csv(Path(load_folder, "annotations", cancers, "embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    mutations_embeddings = pd.read_csv(Path(load_folder, "mutation_embeddings.csv"))

    return sentence_embeddings, image_embeddings, mutations_embeddings


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
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=200000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1, help="Ratio of random noise vectors to add")
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    amount_of_summed_embeddings = args.amount_of_summed_embeddings
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio
    selected_cancers = args.cancer

    cancers = "_".join(selected_cancers)
    save_folder = Path(save_folder, str(amount_of_summed_embeddings), str(noise_ratio))

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    rna_embeddings = load_rna_embeddings(cancers=cancers)
    sentence_embeddings, image_embeddings, mutation_embeddings = load_embeddings(cancers=cancers)

    # drop submitter_id	cancer_type	tile_pos columns from image_embeddings
    image_embeddings.drop(columns=["submitter_id", "cancer_type", "tile_pos"], inplace=True)
    # drop submitter_id column from sentence_embeddings
    sentence_embeddings.drop(columns=["submitter_id"], inplace=True)
    # drop submitter_id and cancer column from rna_embedding
    rna_embeddings.drop(columns=["submitter_id", "cancer"], inplace=True)
    # drop submitter_id column from mutation_embeddings
    mutation_embeddings.drop(columns=["submitter_id"], inplace=True)

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(amount_of_summed_embeddings)):
        combined_sum = pd.Series(np.zeros_like(rna_embeddings.iloc[0]), index=rna_embeddings.columns)
        modality_choices = ['RNA', 'Text', 'Image', 'Mutation']
        combination_counts = {'Text': 0, 'Image': 0, 'RNA': 0, 'Mutation': 0}
        total_noise = True  # Flag to check if only noise was added

        # Randomly allocate the walk_distance across modalities
        modality_counts = dict.fromkeys(modality_choices, 0)
        remaining = walk_distance
        while remaining > 0:
            modality = random.choice(modality_choices)
            modality_counts[modality] += 1
            remaining -= 1

        for modality, count in modality_counts.items():
            if count == 0:
                continue  # Skip modalities with no allocation

            if modality == 'RNA':
                embeddings = rna_embeddings
            elif modality == 'Text':
                embeddings = sentence_embeddings
            elif modality == 'Image':
                embeddings = image_embeddings
            elif modality == 'Mutation':
                embeddings = mutation_embeddings

            # Decide if we should add noise
            add_noise = random.random() < noise_ratio
            current_sum, actual_count, is_noise_only = random_sum_embeddings(embeddings, count, add_noise=add_noise)

            combined_sum += current_sum
            combination_counts[modality] += actual_count
            if not is_noise_only:
                total_noise = False  # At least one real embedding was used

        # If total noise was added, set all combination counts to 0
        if total_noise:
            combination_counts = {'Text': 0, 'Image': 0, 'RNA': 0, 'Mutation': 0}

        # Combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + [combination_counts['Text'], combination_counts['Image'],
                                                   combination_counts['RNA'], combination_counts['Mutation']])

    # Define column names using the columns from one of the embeddings
    column_names = list(rna_embeddings.columns) + ["Text", "Image", "RNA", "Mutation"]

    # Create DataFrame after the loop
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    combined_df = combined_df.astype(float)  # Convert all columns to float

    # Print a message and save the combined embeddings to CSV
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
