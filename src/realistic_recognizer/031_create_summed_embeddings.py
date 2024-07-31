import pandas as pd
from pathlib import Path
import argparse
import numpy as np

save_folder = Path("results", "realistic_recognizer", "embeddings")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer

    cancers = "_".join(selected_cancers)

    # load mappings
    mappings = pd.read_csv(Path("results", "realistic_recognizer", "mappings", "realistic_mappings.csv"))
    # load embeddings
    annotation_embeddings = pd.read_csv(
        Path("results", "realistic_recognizer", "embeddings", "annotations_embeddings.csv"))

    # load cancer embeddings
    cancer_embeddings = {}

    for cancer in selected_cancers:
        cancer_embeddings = pd.read_csv(
            Path("results", "realistic_recognizer", "embeddings", cancers,
                 f"{cancer.lower()}_embeddings.csv"))

    summed_embeddings = []

    for submitter_id in mappings["submitter_id"]:
        concatenated_summed_embeddings = []
        for walk in range(3):
            # Get the annotation embedding
            annotation_embedding = annotation_embeddings[annotation_embeddings["submitter_id"] == submitter_id]

            # Get the cancer type
            cancer = mappings[mappings["submitter_id"] == submitter_id]["cancer"].values[0]

            # Get the cancer embedding
            cancer_embedding = cancer_embeddings[cancer_embeddings["submitter_id"] == submitter_id]

            # Get a random number of text embeddings between 1 and 3
            num_text_embeddings = np.random.randint(1, 4)
            text_embeddings = annotation_embedding.sample(n=num_text_embeddings)

            # Sum all embeddings
            summed_embedding = cancer_embedding.iloc[:, 1:].values + text_embeddings.iloc[:, 1:].sum().values

            # Flatten the summed embedding to a long vector and add to the list for concatenation
            concatenated_summed_embeddings.append(summed_embedding.flatten())

        # Concatenate all three summed embeddings to form a long vector of length 2304
        concatenated_summed_embeddings = np.concatenate(concatenated_summed_embeddings)

        # Create a DataFrame for the concatenated summed embedding with additional columns
        concatenated_embedding_df = pd.DataFrame([concatenated_summed_embeddings])
        concatenated_embedding_df['submitter_id'] = submitter_id
        concatenated_embedding_df['cancer'] = cancer
        concatenated_embedding_df['num_cancer_embeddings'] = len(cancer_embedding)
        concatenated_embedding_df['num_text_embeddings'] = num_text_embeddings

        # Append the concatenated summed embedding DataFrame to the list
        summed_embeddings.append(concatenated_embedding_df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    summed_embeddings = pd.concat(summed_embeddings, ignore_index=True)
    # Save the summed embeddings
    summed_embeddings.to_csv(Path(save_folder, "summed_embeddings.csv"), index=False)
