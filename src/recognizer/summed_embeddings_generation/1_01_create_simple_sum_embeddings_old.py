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


def add_random_or_real_embedding(embeddings, add_noise, count):
    """
    Adds either random Gaussian noise or real embeddings based on add_noise flag.
    """
    combined_sum = pd.Series(np.zeros_like(embeddings.iloc[0]), index=embeddings.columns)
    for _ in range(count):
        if add_noise:
            # Generate random Gaussian noise embedding
            noise_embedding = pd.Series(np.random.normal(0, 1, size=embeddings.shape[1]), index=embeddings.columns)
            combined_sum += noise_embedding
        else:
            # Select a random real embedding
            chosen_index = random.randint(0, len(embeddings) - 1)
            combined_sum += embeddings.iloc[chosen_index]
    return combined_sum


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

    # drop unnecessary columns
    image_embeddings.drop(columns=["submitter_id", "cancer_type", "tile_pos"], inplace=True)
    sentence_embeddings.drop(columns=["submitter_id"], inplace=True)
    rna_embeddings.drop(columns=["submitter_id", "cancer"], inplace=True)
    mutation_embeddings.drop(columns=["submitter_id"], inplace=True)

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(amount_of_summed_embeddings)):
        combined_sum = pd.Series(np.zeros_like(rna_embeddings.iloc[0]), index=rna_embeddings.columns)
        modality_choices = ['RNA', 'Text', 'Image', 'Mutation']
        combination_counts = {'Text': 0, 'Image': 0, 'RNA': 0, 'Mutation': 0}

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

            # Decide whether to add noise or real embeddings
            for _ in range(count):
                add_noise = random.random() < noise_ratio
                if add_noise:
                    # Generate random Gaussian noise embedding
                    noise_embedding = pd.Series(np.random.normal(0, 1, size=embeddings.shape[1]),
                                                index=embeddings.columns)
                    combined_sum += noise_embedding
                else:
                    # Select a random real embedding
                    chosen_index = random.randint(0, len(embeddings) - 1)
                    combined_sum += embeddings.iloc[chosen_index]
                    combination_counts[modality] += 1  # Only count real embeddings

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
