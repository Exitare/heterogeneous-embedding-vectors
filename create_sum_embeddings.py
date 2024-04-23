import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path


def load_embeddings():
    # Load the embeddings from CSV files
    rna_embeddings = pd.read_csv("results/embeddings/rna_embeddings.csv")
    sentence_embeddings = pd.read_csv("results/embeddings/sentence_embeddings.csv")
    image_embeddings = pd.read_csv("results/embeddings/image_embeddings.csv")
    return rna_embeddings, sentence_embeddings, image_embeddings


def random_sum_embeddings(embeddings):
    # Randomly choose 1 to 3 embeddings and sum them
    n = random.randint(0, 3)  # Random count of embeddings to sum
    chosen_indices = random.sample(range(len(embeddings)), n)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, n


if __name__ == '__main__':
    # Load embeddings
    rna_embeddings, sentence_embeddings, image_embeddings = load_embeddings()

    # List to hold all combined embeddings and their indices
    combined_data = []

    for _ in tqdm(range(20000)):  # Generate 10,000 combined embeddings
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
    column_names = list(rna_embeddings.columns) + ["Text", "Image", "RNA"]  # Assume all embeddings have the same columns
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    # Save the new embeddings
    combined_df.to_csv(Path("results","summed_embeddings.csv"), index=False)
