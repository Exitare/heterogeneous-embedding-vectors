import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_folder = Path("results", "recognizer_embeddings")
cancer_embedding_load_folder = Path(load_folder, "cancer")
save_folder = Path("results", "recognizer", "summed_embeddings", "multi")


def load_noise_embeddings():
    # Load the embeddings from CSV files
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, count):
    # Select exactly 'count' number of embeddings
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
    args = parser.parse_args()

    iterations = args.iterations
    walk_distance = args.walk_distance
    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)

    cancer_embedding_load_folder = Path(cancer_embedding_load_folder, cancers)
    save_folder = Path(save_folder, str(iterations), cancers)
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

    # drop submitter_id	cancer_type	tile_pos columns from image_embeddings
    image_embeddings.drop(columns=["submitter_id", "cancer_type", "tile_pos"], inplace=True)

    # Prepare a list of all available modalities dynamically
    modality_names = ['Text', 'Image'] + list(loaded_cancer_embeddings.keys())
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Create dynamic list of embeddings with modality names
        embeddings_list = [(sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        for cancer_type, cancer_embedding in loaded_cancer_embeddings.items():
            embeddings_list.append((cancer_embedding, cancer_type))

        embeddings_list.append((sentence_embeddings, 'Text'))
        embeddings_list.append((sentence_embeddings, 'Text'))
        embeddings_list.append((image_embeddings, 'Image'))
        embeddings_list.append((image_embeddings, 'Image'))

        random.shuffle(embeddings_list)

        # Initialize combined sum and remaining embeddings counter
        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = walk_distance
        combination_counts = {name: 0 for name in modality_names}

        # assuming embeddings_list and remaining_embeddings are defined elsewhere
        for i, (embeddings, name) in enumerate(embeddings_list):
            if remaining_embeddings == 0:
                break

            # If it's the last item in the embeddings_list, we need to ensure the count matches exactly
            if i == len(embeddings_list) - 1:
                max_embeddings_for_type = remaining_embeddings
            else:
                # Select a random number between 0 and the remaining embeddings
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

            # Check if the remaining_embeddings goes negative, which shouldn't happen
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

    # convert TExt, image cancers and RNA to int
    combined_df["Text"] = combined_df["Text"].astype(int)
    combined_df["Image"] = combined_df["Image"].astype(int)
    combined_df["RNA"] = combined_df["RNA"].astype(int)

    for cancer in loaded_cancer_embeddings.keys():
        combined_df[cancer] = combined_df[cancer].astype(int)

    print(combined_df)

    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{walk_distance}_embeddings.csv"), index=False)
