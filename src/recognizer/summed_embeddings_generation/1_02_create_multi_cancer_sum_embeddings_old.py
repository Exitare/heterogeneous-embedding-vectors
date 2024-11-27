import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_folder = Path("results", "embeddings")
cancer_embedding_load_folder = Path(load_folder, "rna")
save_folder = Path("results", "recognizer", "summed_embeddings", "multi")


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


def random_sum_embeddings(embeddings, count):
    # Select exactly 'count' number of embeddings
    chosen_indices = random.sample(range(len(embeddings)), count)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, count


def generate_noise(embedding_length, scale=0.1):
    return np.random.normal(loc=0, scale=scale, size=embedding_length)


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=200000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True,
                        help="The selected cancer identifier to sum")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.1, help="Ratio of random noise vectors to add")
    args = parser.parse_args()

    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    walk_distance: int = args.walk_distance
    noise_ratio: float = args.noise_ratio
    selected_cancers: [] = args.selected_cancers
    cancers = "_".join(selected_cancers)

    cancer_embedding_load_folder = Path(cancer_embedding_load_folder, cancers)
    save_folder = Path(save_folder, cancers, str(amount_of_summed_embeddings), str(noise_ratio))
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    print(f"Using save folder: {save_folder}")

    # Dynamically load all embeddings
    loaded_rna_embeddings = {}
    for cancer in selected_cancers:
        try:
            temp_df = pd.read_csv(Path(cancer_embedding_load_folder, f"{cancer.lower()}_embeddings.csv"))
            # remove patient column if exists
            if "Patient" in temp_df.columns:
                temp_df.drop(columns=["Patient"], inplace=True)
            loaded_rna_embeddings[cancer] = temp_df
        except FileNotFoundError:
            print(f"Could not load {cancer} embedding. Skipping this cancer type.")
            continue

    if not loaded_rna_embeddings:
        raise ValueError("No valid cancer embeddings were loaded. Please check your input.")

    # Load embeddings
    sentence_embeddings, image_embeddings, mutation_embeddings = load_embeddings(cancers=cancers)

    # Drop unnecessary columns
    image_embeddings.drop(columns=["submitter_id", "cancer_type", "tile_pos"], inplace=True)
    sentence_embeddings.drop(columns=["submitter_id"], inplace=True)
    loaded_rna_embeddings = {cancer: df.drop(columns=["submitter_id", "cancer"]) for cancer, df in
                             loaded_rna_embeddings.items()}
    mutation_embeddings.drop(columns=["submitter_id"], inplace=True)

    # Prepare a list of all available modalities dynamically
    modality_names = ['Text', 'Image', "Mutation"] + list(loaded_rna_embeddings.keys())
    combined_data = []

    for _ in tqdm(range(amount_of_summed_embeddings)):
        # Create dynamic list of embeddings with modality names
        embeddings_list = [(sentence_embeddings, 'Text'),
                           (image_embeddings, 'Image'), (mutation_embeddings, 'Mutation')]
        for cancer_type, cancer_embedding in loaded_rna_embeddings.items():
            embeddings_list.append((cancer_embedding, cancer_type))

        random.shuffle(embeddings_list)

        # Initialize combined sum and remaining embeddings counter
        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = walk_distance
        combination_counts = {name: 0 for name in modality_names}

        for i, (embeddings, name) in enumerate(embeddings_list):
            if remaining_embeddings == 0:
                break

            # Add noise if specified and probability condition matches
            if random.random() < noise_ratio:
                noise_vector = generate_noise(len(embeddings.columns))
                combined_sum += noise_vector
                remaining_embeddings -= 1
                continue

            # If it's the last item in the embeddings_list, ensure the count matches exactly
            if i == len(embeddings_list) - 1:
                max_embeddings_for_type = remaining_embeddings
            else:
                max_embeddings_for_type = random.randint(0, remaining_embeddings)

            # Select the random sum embedding
            current_sum, count = random_sum_embeddings(embeddings, max_embeddings_for_type)

            # Update the combined sum
            combined_sum += current_sum

            # Update the remaining embeddings
            remaining_embeddings -= count

            # Update the combination counts
            if name in combination_counts:
                combination_counts[name] += count
            else:
                combination_counts[name] = count

            # Validate remaining embeddings
            if remaining_embeddings < 0:
                raise ValueError(
                    "Selected more embeddings than remaining. Check the random_sum_embeddings function or logic.")

        # Append combined sum and counts for each modality dynamically
        combined_row = list(combined_sum) + [combination_counts[modality] for modality in modality_names]
        combined_data.append(combined_row)

    # Define column names dynamically for CSV export
    column_names = list(embeddings_list[0][0].columns) + modality_names

    combined_df = pd.DataFrame(combined_data, columns=column_names)

    # Add RNA column as the sum of all cancer types in the dataset
    combined_df["RNA"] = combined_df[list(loaded_rna_embeddings.keys())].sum(axis=1)
    # Convert all columns to float for consistency
    combined_df = combined_df.astype(float)

    # Convert Text, Image, Mutation, and RNA counts to int
    combined_df["Text"] = combined_df["Text"].astype(int)
    combined_df["Image"] = combined_df["Image"].astype(int)
    combined_df["RNA"] = combined_df["RNA"].astype(int)
    combined_df["Mutation"] = combined_df["Mutation"].astype(int)

    for cancer in loaded_rna_embeddings.keys():
        combined_df[cancer] = combined_df[cancer].astype(int)

    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
