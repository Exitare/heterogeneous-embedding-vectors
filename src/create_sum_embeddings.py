import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

load_folder = Path("results", "embeddings")
save_folder = Path("results")


def load_embeddings():
    # Load the embeddings from CSV files
    rna_embeddings = pd.read_csv(Path(load_folder, "rna_embeddings.csv"))
    sentence_embeddings = pd.read_csv(Path(load_folder, "sentence_embeddings.csv"))
    image_embeddings = pd.read_csv(Path(load_folder, "image_embeddings.csv"))
    return rna_embeddings, sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings):
    # Randomly choose 1 to 3 embeddings and sum them
    n = random.randint(0, 3)  # Random count of embeddings to sum
    chosen_indices = random.sample(range(len(embeddings)), n)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, n


if __name__ == '__main__':

    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--embeddings", "-e", type=Path, nargs='+')
    parser.add_argument("--iterations", "-i", type=int, default=200000, help="Number of iterations to run")
    args = parser.parse_args()

    iterations = args.iterations
    embeddings = args.embeddings

    # Load embeddings
    rna_embeddings, sentence_embeddings, image_embeddings = load_embeddings()

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(iterations)):
        # Randomly sum embeddings for each type independently
        rna_sum, rna_count = random_sum_embeddings(rna_embeddings)
        sentence_sum, sentence_count = random_sum_embeddings(sentence_embeddings)
        image_sum, image_count = random_sum_embeddings(image_embeddings)

        # Sum the embeddings from each type
        combined_sum = rna_sum + sentence_sum + image_sum

        # create columns for each count separately
        combination_counts = [
            sentence_count,
            image_count,
            rna_count
        ]

        # combine combined_sum and the combination_counts
        combined_data.append(list(combined_sum) + combination_counts)

    # Create DataFrame from combined data
    column_names = list(rna_embeddings.columns) + ["Text", "Image",
                                                   "RNA"]  # Assume all embeddings have the same columns
    combined_df = pd.DataFrame(combined_data, columns=column_names)

    # Save the new embeddings
    print("Saving combined embeddings to CSV...")
    combined_df.to_csv(Path(save_folder, "summed_embeddings.csv"), index=False)
