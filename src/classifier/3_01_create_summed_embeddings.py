import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from matplotlib.style import available

save_folder = Path("results", "classifier", "summed_embeddings")


# Function to perform a random walk and select embeddings
def random_walk_selection(patient_id, patient_cancer_embedding, patient_annotations, patient_mutations, walk_distance):
    # Initialize the selected embeddings dictionary
    selected_embeddings = {
        'cancer': [],
        'text': [],
        'mutation': []
    }

    # Create a dictionary of available embeddings, which are not empty
    available_nodes = {}
    if not patient_cancer_embedding.empty:
        available_nodes['cancer'] = patient_cancer_embedding
    if not patient_annotations.empty:
        available_nodes['text'] = patient_annotations
    if not patient_mutations.empty:
        available_nodes['mutation'] = patient_mutations

    selected_count = 0

    # Perform the random walk until the exact number of embeddings is selected
    while selected_count < walk_distance:
        # Randomly choose a type of node to visit
        chosen_type = np.random.choice(list(available_nodes.keys()))
        available_embeddings = available_nodes[chosen_type]

        # If there are any embeddings of the chosen type left to select
        if not available_embeddings.empty:
            # Select a random embedding and add it to the selected embeddings
            selected_embedding = available_embeddings.sample(n=1)
            selected_embeddings[chosen_type].append(selected_embedding)

            # Increment the count of selected embeddings
            selected_count += 1

    # Convert lists of embeddings into dataframes
    for key in selected_embeddings.keys():
        if selected_embeddings[key]:  # If there are any embeddings selected
            selected_embeddings[key] = pd.concat(selected_embeddings[key])
        else:  # If no embeddings were selected
            selected_embeddings[key] = pd.DataFrame()

    return selected_embeddings


def sum_embeddings(selected_embeddings):
    # Initialize a sum_embedding DataFrame with the same columns as the embeddings, filled with zeros
    # This assumes that all embeddings have the same set of columns
    embedding_sum = None

    for modality, embeddings in selected_embeddings.items():
        if not embeddings.empty:
            if embedding_sum is None:
                # Initialize the sum_embedding DataFrame on the first encounter of non-empty embeddings
                embedding_sum = embeddings.sum()
            else:
                # Sum embeddings of the current modality
                embedding_sum += embeddings.sum()

    # Convert the summed embeddings to a DataFrame, if not already done
    if embedding_sum is not None:
        embedding_sum = pd.DataFrame(embedding_sum).transpose()

    return embedding_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5], default=3)
    args = parser.parse_args()

    selected_cancers = args.cancer
    walk_distance = args.walk_distance
    amount_of_walks = args.amount_of_walks
    print("Selected cancers: ", selected_cancers)
    print(f"Using walk distance of {walk_distance} and {amount_of_walks} walks.")

    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, f"{walk_distance}_{amount_of_walks}")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load mappings
    mappings = pd.read_csv(Path("results", "classifier", "mappings", cancers, "mappings.csv"))
    # load embeddings
    text_annotation_embeddings = pd.read_csv(
        Path("results", "classifier", "embeddings", "annotations", cancers, "embeddings.csv"))

    # load mutation embeddings
    mutation_embeddings = pd.read_csv(
        Path("results", "classifier", "embeddings", "mutation_embeddings.csv"))

    # find all submitter ids with only 1 annotation
    single_annotation = text_annotation_embeddings["submitter_id"].value_counts()[
        text_annotation_embeddings["submitter_id"].value_counts() == 1].index
    if len(single_annotation) != 0:
        print(single_annotation)
        print(f"Number of patients with only 1 annotation: {len(single_annotation)}")

    # check that each text annotation per patient has more than 1 entry
    assert text_annotation_embeddings[
               "submitter_id"].value_counts().min() > 1, "Each patient should have more than 1 text annotation"

    # load cancer embeddings
    cancer_embeddings = {}

    print("Loading cancer embeddings...")
    for cancer in selected_cancers:
        cancer_embeddings[cancer] = pd.read_csv(
            Path("results", "classifier", "embeddings", "annotated_cancer", cancers,
                 f"{cancer.lower()}_embeddings.csv"))

    summed_embeddings = []

    print("Creating summed embeddings...")
    for patient_id in mappings["patient"]:
        row = mappings[mappings["patient"] == patient_id]

        patient_mutation_id = row["mutation_id"].values[0]
        # load patient mutations
        patient_mutations = mutation_embeddings[mutation_embeddings["submitter_id"] == patient_mutation_id]
        patient_mutations = patient_mutations.drop(columns=["submitter_id"])

        patient_annotation_id = row["submitter_id"].values[0]
        # Get the annotation embedding
        patient_annotations = text_annotation_embeddings[
            text_annotation_embeddings["submitter_id"] == patient_annotation_id]
        # drop submitter_id from text_embeddings
        patient_annotations = patient_annotations.drop(columns=["submitter_id"])

        # Get the cancer type
        cancer_type = mappings[mappings["patient"] == patient_id]["cancer"].values[0]

        # Get the cancer embedding
        cancer_specific_embeddings = cancer_embeddings[cancer_type]
        patient_cancer_embedding = cancer_specific_embeddings[cancer_specific_embeddings["patient"] == patient_id]
        # drop submitter and patient from cancer_embedding
        patient_cancer_embedding = patient_cancer_embedding.drop(columns=["submitter_id", "patient"])

        concatenated_summed_embeddings = []
        for walk in range(amount_of_walks):
            # Call the function with the provided data
            selected_embeddings = random_walk_selection(patient_id, patient_cancer_embedding, patient_annotations,
                                                        patient_mutations, walk_distance)

            # Summing all embeddings
            summed_embedding = sum_embeddings(selected_embeddings)

            # assert that shape is 768 columns
            assert summed_embedding.shape[
                       1] == 768, f"Shape of summed embedding should be 768 columns, {summed_embedding.shape})"
            # assert that there is only one row
            assert summed_embedding.shape[
                       0] == 1, f"Shape of summed embedding should be 1 row, {summed_embedding.shape}, {patient_id}"

            concatenated_summed_embeddings.append(summed_embedding)

        # assert that length of concatenated_summed_embeddings is 3
        assert len(concatenated_summed_embeddings) == amount_of_walks, (
            f"The length of the concatenated summed embeddings should be 3, {len(concatenated_summed_embeddings)}")

        # assert that each element in the concatenated_summed_embeddings is a numpy array with 768 columns
        for element in concatenated_summed_embeddings:
            assert element.shape[1] == 768, (
                f"Each element in the concatenated summed embeddings should have 768 columns, {element.shape}")

        # Concatenate all three summed embeddings to form a long vector of length 2304
        concatenated_summed_embeddings = np.concatenate(concatenated_summed_embeddings).flatten()

        # assert that length of concatenated_summed_embeddings is 2306
        assert len(
            concatenated_summed_embeddings) == 768 * amount_of_walks, (
            f"The length of the concatenated summed embedding should be 2304, "
            f"{len(concatenated_summed_embeddings)}")

        # Create a DataFrame for the concatenated summed embedding with additional columns
        concatenated_embedding_df = pd.DataFrame([concatenated_summed_embeddings])
        concatenated_embedding_df['patient_id'] = patient_id
        concatenated_embedding_df['cancer'] = cancer_type

        # assert that length of concatenated_embedding_df is 2306
        assert concatenated_embedding_df.shape[
                   1] == 768 * amount_of_walks + 2, f"The length of the concatenated embedding should be {768 * amount_of_walks + 2}, but is {concatenated_embedding_df.shape}"

        # convert the concatenated_embedding_df to a dictionary
        concatenated_embedding_df = concatenated_embedding_df.to_dict(orient="records")[0]

        # Append the concatenated summed embedding DataFrame to the list
        summed_embeddings.append(concatenated_embedding_df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    summed_embeddings = pd.DataFrame(summed_embeddings)

    # assert that all selected cancers are in the cancer column
    assert all(cancer in summed_embeddings["cancer"].unique() for cancer in selected_cancers), (
        f"All selected cancers should be in the summed embeddings, {summed_embeddings['cancer'].unique()}")

    print(summed_embeddings)
    print("Summed embeddings created.")
    print("Saving summed embeddings...")
    # Save the summed embeddings
    summed_embeddings.to_csv(Path(save_folder, "summed_embeddings.csv"), index=False)
    print("Summed embeddings saved.")
