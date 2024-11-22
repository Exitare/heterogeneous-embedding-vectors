import h5py
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

save_folder = Path("results", "recognizer", "summed_embeddings", "test")
load_file = Path("embedding_data.h5")


def get_total_rows_and_columns(file_path, group):
    """
    Returns the total number of rows and ensures columns are compatible for summation.
    """
    with h5py.File(file_path, 'r') as f:
        dataset = f[group]["embeddings"]
        total_rows = dataset.shape[0]
        total_columns = min(len(dataset.dtype), 767)  # Limit to 767 columns
    return total_rows, total_columns


def get_random_row_from_hdf5(file_path, group, num_rows):
    """
    Efficiently retrieves a random row from a specific group in the HDF5 file and slices to 767 columns.
    """
    random_idx = random.randint(0, num_rows - 1)
    with h5py.File(file_path, 'r') as f:
        dataset = f[group]["embeddings"]
        row = dataset[random_idx]
        # Convert structured row to a dict, then a pandas Series, and ensure only the first 767 columns are used
        row_as_dict = {name: row[name] for name in row.dtype.names[:767]}  # Use only first 767 columns
    return pd.Series(row_as_dict).astype(float)  # Convert to numeric


def add_random_or_real_embedding(group_name, total_rows, num_columns, add_noise):
    """
    Adds either random Gaussian noise or a real embedding from a specific modality group.
    """
    if add_noise:
        # Generate random Gaussian noise embedding
        noise_embedding = pd.Series(np.random.normal(0, 1, size=num_columns))
        return noise_embedding
    else:
        # Select a random real embedding from the group
        random_row = get_random_row_from_hdf5(load_file, group_name, total_rows)
        return random_row


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum")
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=200000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1, help="Ratio of random noise vectors to add")
    args = parser.parse_args()

    amount_of_summed_embeddings = args.amount_of_summed_embeddings
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio

    save_folder = Path(save_folder, str(amount_of_summed_embeddings), str(noise_ratio))
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Get the total number of rows and columns for each modality from the HDF5 file
    rna_total_rows, rna_columns = get_total_rows_and_columns(load_file, "rna")
    image_total_rows, image_columns = get_total_rows_and_columns(load_file, "images")
    mutation_total_rows, mutation_columns = get_total_rows_and_columns(load_file, "mutations")
    annotation_total_rows, annotation_columns = get_total_rows_and_columns(load_file, "annotations")

    # Ensure all modalities have the same column dimensions
    assert rna_columns == image_columns == mutation_columns == annotation_columns == 767, \
        "All modalities must have exactly 767 usable columns for summation"

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(amount_of_summed_embeddings)):
        # Initialize the combined sum with zeros for the embedding dimensions
        combined_sum = pd.Series(np.zeros(767))  # Assuming all modalities share the same dimensions
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

            for _ in range(count):
                add_noise = random.random() < noise_ratio
                if modality == 'RNA':
                    embedding = add_random_or_real_embedding('rna', rna_total_rows, rna_columns, add_noise)
                elif modality == 'Text':
                    embedding = add_random_or_real_embedding('annotations', annotation_total_rows, annotation_columns,
                                                             add_noise)
                elif modality == 'Image':
                    embedding = add_random_or_real_embedding('images', image_total_rows, image_columns, add_noise)
                elif modality == 'Mutation':
                    embedding = add_random_or_real_embedding('mutations', mutation_total_rows, mutation_columns,
                                                             add_noise)

                # Ensure indices match and convert to float
                embedding = embedding.reset_index(drop=True).astype(float)
                combined_sum = combined_sum.reset_index(drop=True).astype(float)

                # Add the embedding to the combined sum
                combined_sum += embedding

                # Only count real embeddings
                if not add_noise:
                    combination_counts[modality] += 1

        # Combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + [combination_counts['Text'], combination_counts['Image'],
                                                   combination_counts['RNA'], combination_counts['Mutation']])

    # Define column names using the dimensions from one of the embeddings
    column_names = [f"dim_{i}" for i in range(767)] + ["Text", "Image", "RNA", "Mutation"]

    # Create DataFrame after the loop
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    combined_df = combined_df.astype(float)  # Convert all columns to float

    # Print a message and save the combined embeddings to CSV
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
