import h5py
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_file = Path("embedding_data.h5")
save_folder = Path("results", "recognizer", "summed_embeddings", "multi")
dims = 767

def get_total_rows_and_columns(file_path, group):
    with h5py.File(file_path, 'r') as f:
        dataset = f[group]["embeddings"]
        total_rows = dataset.shape[0]
        total_columns = min(len(dataset.dtype), dims)  # Limit to dims columns
    return total_rows, total_columns

def filter_rows_by_cancer(file_path, group, cancer_type):
    with h5py.File(file_path, 'r') as f:
        dataset = f[group]["embeddings"]
        indices = [i for i, row in enumerate(dataset) if row["cancer"].decode("utf-8") == cancer_type]
    return indices

def random_sum_embeddings(file_path, group, indices, count):
    chosen_indices = random.sample(indices, count)
    with h5py.File(file_path, 'r') as f:
        dataset = f[group]["embeddings"]
        chosen_rows = [dataset[idx] for idx in chosen_indices]
        chosen_sums = pd.DataFrame.from_records(chosen_rows).iloc[:, :dims].sum(axis=0)
    return chosen_sums.astype(float), count

def generate_noise(embedding_length, scale=0.1):
    return np.random.normal(loc=0, scale=scale, size=embedding_length)

if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=200000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True,
                        help="The selected cancer identifiers to sum")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1, help="Ratio of random noise vectors to add")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    amount_of_summed_embeddings = args.amount_of_summed_embeddings
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio
    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)
    debug = args.debug

    if debug:
        print(f"Selected cancers: {selected_cancers}")
        print(f"Walk distance: {walk_distance}")
        print(f"Amount of summed embeddings: {amount_of_summed_embeddings}")
        print(f"Noise ratio: {noise_ratio}")

    save_folder = Path(save_folder, cancers, str(amount_of_summed_embeddings), str(noise_ratio))
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    if debug:
        print(f"Using save folder: {save_folder}")

    # Validate available indices for each selected cancer
    cancer_indices = {}
    for cancer in selected_cancers:
        indices = filter_rows_by_cancer(load_file, "rna", cancer)
        if len(indices) == 0:
            print(f"Warning: No rows found for cancer type '{cancer}'. Skipping.")
        else:
            cancer_indices[cancer] = indices
            if debug:
                print(f"Found {len(indices)} rows for cancer type '{cancer}'.")

    if len(cancer_indices) == 0:
        raise ValueError("No valid cancer types found. Exiting.")

    # Load total rows for other modalities
    annotation_total_rows, _ = get_total_rows_and_columns(load_file, "annotations")
    image_total_rows, _ = get_total_rows_and_columns(load_file, "images")
    mutation_total_rows, _ = get_total_rows_and_columns(load_file, "mutations")

    # Prepare combined data
    modality_names = ["Text", "Image", "Mutation"] + list(cancer_indices.keys()) + ["RNA"]
    combined_data = []

    for _ in tqdm(range(amount_of_summed_embeddings)):
        # Randomly select one cancer type
        selected_cancer = random.choice(selected_cancers)
        rna_data = cancer_indices[selected_cancer]

        embeddings_list = [
            ("annotations", list(range(annotation_total_rows)), "Text"),
            ("images", list(range(image_total_rows)), "Image"),
            ("mutations", list(range(mutation_total_rows)), "Mutation"),
            ("rna", rna_data, selected_cancer),
        ]

        combined_sum = pd.Series(np.zeros(dims))
        remaining_embeddings = walk_distance
        combination_counts = {modality: 0 for modality in modality_names}

        for _ in range(walk_distance):
            if random.random() < noise_ratio:
                # Add noise vector with a small probability
                noise_vector = generate_noise(dims)
                combined_sum += noise_vector
                continue

            # Randomly pick from modalities, prioritizing the selected cancer for RNA
            group, data, name = random.choice(embeddings_list)
            if group == "rna" and name != selected_cancer:
                continue  # Ensure only the selected cancer's RNA is used

            current_sum, used_count = random_sum_embeddings(load_file, group, data, 1)
            combined_sum += current_sum
            combination_counts[name] += 1

        # Ensure RNA column reflects the count of the selected cancer
        combination_counts["RNA"] = combination_counts[selected_cancer]

        combined_row = list(combined_sum) + [combination_counts[modality] for modality in modality_names]
        combined_data.append(combined_row)

    column_names = [i for i in range(dims)] + modality_names
    combined_df = pd.DataFrame(combined_data, columns=column_names)

    # Convert all columns to float for consistency
    combined_df = combined_df.astype(float)

    # Convert Text, Image, Mutation, and RNA counts to int
    combined_df["Text"] = combined_df["Text"].astype(int)
    combined_df["Image"] = combined_df["Image"].astype(int)
    combined_df["RNA"] = combined_df["RNA"].astype(int)
    combined_df["Mutation"] = combined_df["Mutation"].astype(int)

    for cancer in cancer_indices.keys():
        combined_df[cancer] = combined_df[cancer].astype(int)

    if debug:
        # Print only the modalities columns, which are the last columns
        print(combined_df[modality_names].head())

    for col in combined_df.columns:
        try:
            combined_df[col] = combined_df[col].astype(float)
        except Exception as e:
            print(f"Error converting column {col} to float: {e}")

    combined_df.to_hdf(Path(save_folder, f"{walk_distance}_embeddings.h5"), key="embeddings", mode="w", format="table")
    print(f"Saved combined embeddings to {save_folder}.")