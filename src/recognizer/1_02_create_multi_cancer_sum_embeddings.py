import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

load_folder = Path("results", "recognizer", "embeddings")
cancer_embedding_load_folder = Path(load_folder, "cancer")
save_folder = Path("results", "recognizer", "summed_embeddings", "multi")

def validate_single_cancer_type_per_row(df, cancer_columns):
    for index, row in df.iterrows():
        # Check if there's any non-zero RNA value
        if row['RNA'] > 0:
            non_zero_count = row[cancer_columns].astype(bool).sum()
            if non_zero_count != 1:
                raise ValueError(f"Row {index} does not have exactly one non-zero cancer type: {row[cancer_columns].to_dict()}")



def load_noise_embeddings():
    # Load the embeddings from CSV files
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings, max_count):
    # Randomly choose embeddings up to max_count
    n = random.randint(1, max_count)  # Ensure at least one is selected
    chosen_indices = random.sample(range(len(embeddings)), n)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, n


if __name__ == '__main__':
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--embeddings", "-e", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True,
                        help="The selected cancer identifier to sum")
    args = parser.parse_args()

    iterations = args.iterations
    total_embeddings = args.embeddings
    selected_cancers = args.selected_cancers
    cancers = "_".join(selected_cancers)

    cancer_embedding_load_folder = Path(cancer_embedding_load_folder, cancers)
    save_folder = Path(save_folder, cancers)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    print(f"Using save folder: {save_folder}")

    loaded_cancer_embeddings = {}
    for cancer in selected_cancers:
        try:
            temp_df = pd.read_csv(Path(cancer_embedding_load_folder, f"{cancer.lower()}_embeddings.csv"))
            # Remove patient column if exists
            if "Patient" in temp_df.columns:
                temp_df.drop(columns=["Patient"], inplace=True)
            cancer_type = cancer
            loaded_cancer_embeddings[cancer_type] = temp_df
        except:
            print(f"Could not load {cancer} embedding...")
            raise

    # Load noise embeddings
    sentence_embeddings, image_embeddings = load_noise_embeddings()

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Select one random cancer type for this iteration
        selected_cancer_type = random.choice(selected_cancers)
        cancer_embedding = loaded_cancer_embeddings[selected_cancer_type]

        # Create a list of tuples with the embeddings and their names
        embeddings_list = [(sentence_embeddings, 'Text'), (image_embeddings, 'Image'),
                           (cancer_embedding, selected_cancer_type)]

        random.shuffle(embeddings_list)

        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = total_embeddings
        combination_counts = {key: 0 for key in ['Image', 'Text'] + list(loaded_cancer_embeddings.keys())}

        for embeddings, name in embeddings_list:
            if remaining_embeddings > 0:
                current_sum, count = random_sum_embeddings(embeddings, remaining_embeddings)
                combined_sum += current_sum
                remaining_embeddings -= count
                combination_counts[name] += count

        # Ensure there is at least one embedding selected in total (avoid all-zero entries)
        if all(v == 0 for v in combination_counts.values()):
            embeddings, name = random.choice(embeddings_list)
            current_sum, count = random_sum_embeddings(embeddings, 1)  # Force at least one selection
            combined_sum += current_sum
            combination_counts[name] += count

        # Sort the combination counts by the keys
        combination_counts = dict(sorted(combination_counts.items()))

        # Combine combined_sum and the combination_counts which are Image, Text and the cancer types
        combined_data.append(list(combined_sum) + [combination_counts['Image'], combination_counts['Text']] + [
            combination_counts[cancer_type] for cancer_type in loaded_cancer_embeddings.keys()])

    # Save the data to CSV
    column_names = list(embeddings_list[0][0].columns) + ['Image', 'Text'] + [
        cancer_type for cancer_type in loaded_cancer_embeddings.keys()]

    combined_df = pd.DataFrame(combined_data, columns=column_names)

    # Create another column called RNA which is the sum of the cancer types in the dataset, defined by the loaded_cancer_embeddings
    combined_df["RNA"] = combined_df[[cancer_type for cancer_type in loaded_cancer_embeddings.keys()]].sum(axis=1)

    # Convert all columns to float
    combined_df = combined_df.astype(float)

    print(combined_df)

    # assert that combined df is 0 for all cancer types except one, but the amount can be anything
    # Perform the validation
    validate_single_cancer_type_per_row(combined_df, selected_cancers)

    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, f"{total_embeddings}_embeddings.csv"), index=False)
