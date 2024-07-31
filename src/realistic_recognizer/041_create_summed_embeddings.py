import pandas as pd
from pathlib import Path
import argparse

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
            summed_embedding = cancer_embedding + text_embeddings.sum()

            # Add the number of cancer embeddings and text embeddings as new columns
            summed_embedding['num_cancer_embeddings'] = len(cancer_embedding)
            summed_embedding['num_text_embeddings'] = num_text_embeddings

            # Assuming you want to store or use the summed_embedding
            # Here, you can append it to a list or DataFrame for further use
            # For example, let's append it to a list:
            summed_embeddings.append(summed_embedding)

    summed_embeddings = pd.DataFrame(summed_embeddings)

    # Save the summed embeddings
    summed_embeddings.to_csv(Path(save_folder, "summed_embeddings.csv"), index=False)
