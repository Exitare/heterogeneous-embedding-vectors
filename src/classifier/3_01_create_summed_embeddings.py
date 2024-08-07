import pandas as pd
from pathlib import Path
import argparse
import numpy as np

save_folder = Path("results", "classifier", "summed_embeddings")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    print("Selected cancers: ", selected_cancers)

    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load mappings
    mappings = pd.read_csv(Path("results", "classifier", "mappings", "realistic_mappings.csv"))
    # load embeddings
    text_annotation_embeddings = pd.read_csv(
        Path("results", "classifier", "embeddings", "annotations_embeddings.csv"))

    # load mutation embeddings
    mutation_embeddings = pd.read_csv(
        Path("results", "classifier", "embeddings", "mutation_embeddings.csv"))

    # find all submitter ids with only 1 annotation
    single_annotation = text_annotation_embeddings["submitter_id"].value_counts()[
        text_annotation_embeddings["submitter_id"].value_counts() == 1].index
    if len(single_annotation == 0):
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
    for submitter_id in mappings["submitter_id"]:
        concatenated_summed_embeddings = []
        patient_mutation_id = mappings[mappings["submitter_id"] == submitter_id]["mutation_id"].values[0]
        for walk in range(3):
            # Get the annotation embedding
            patient_annotations = text_annotation_embeddings[text_annotation_embeddings["submitter_id"] == submitter_id]
            # patient annotations should not be empty
            assert patient_annotations.shape[0] > 0, (
                f"Patient annotations should not be empty, {patient_annotations.shape}, {submitter_id}")

            # load patient mutations
            patient_mutations = mutation_embeddings[mutation_embeddings["submitter_id"] == patient_mutation_id]

            # Get the cancer type
            cancer_type = mappings[mappings["submitter_id"] == submitter_id]["cancer"].values[0]

            # Get the cancer embedding
            cancer_specific_embeddings = cancer_embeddings[cancer_type]
            cancer_embedding = cancer_specific_embeddings[cancer_specific_embeddings["submitter_id"] == submitter_id]

            # if more than one cancer embeddings is found, randomly select one
            if cancer_embedding.shape[0] > 1:
                cancer_embedding = cancer_embedding.sample(n=1)

            # cancer embedding should only have one row
            assert cancer_embedding.shape[0] == 1, (
                f"Cancer embedding should only have one row, {cancer_embedding.shape}, {submitter_id}")

            # select max text embeddings or 4
            max_text_embeddings = 4
            if len(patient_annotations) < 4:
                max_text_embeddings = len(patient_annotations)

            num_text_embeddings = np.random.randint(1, max_text_embeddings)
            text_embeddings = patient_annotations.sample(n=num_text_embeddings)

            if len(patient_mutations) != 0:
                patient_mutations = patient_mutations.sample(n=1)

            # drop submitter_id from text_embeddings
            text_embeddings = text_embeddings.drop(columns=["submitter_id"])
            # drop submitter and patient from cancer_embedding
            cancer_embedding = cancer_embedding.drop(columns=["submitter_id", "patient"])
            # drop submitter_id from mutation_embeddings
            patient_mutations = patient_mutations.drop(columns=["submitter_id"])

            assert "submitter_id" not in text_embeddings.columns
            assert "submitter_id" not in cancer_embedding.columns
            assert "patient" not in cancer_embedding.columns
            assert "submitter_id" not in patient_mutations.columns


            # Sum all embeddings
            if len(patient_mutations) == 0:
                summed_embedding = cancer_embedding.values + text_embeddings.sum().values
            else:
                summed_embedding = cancer_embedding.values + text_embeddings.sum().values + patient_mutations.sum().values

            # assert that shape is 768 columns
            assert summed_embedding.shape[
                       1] == 768, f"Shape of summed embedding should be 768 columns, {summed_embedding.shape})"
            # assert that there is only one row
            assert summed_embedding.shape[
                       0] == 1, f"Shape of summed embedding should be 1 row, {summed_embedding.shape}, {submitter_id}"

            # Flatten the summed embedding to a long vector and add to the list for concatenation
            concatenated_summed_embeddings.append(summed_embedding.flatten())

        # assert that length of concatenated_summed_embeddings is 3
        assert len(concatenated_summed_embeddings) == 3, (
            f"The length of the concatenated summed embeddings should be 3, {len(concatenated_summed_embeddings)}")

        # assert that each element in the concatenated_summed_embeddings is a numpy array with 768 columns
        for element in concatenated_summed_embeddings:
            assert element.shape[0] == 768, (
                f"Each element in the concatenated summed embeddings should have 768 columns, {element.shape}")

        # Concatenate all three summed embeddings to form a long vector of length 2304
        concatenated_summed_embeddings = np.concatenate(concatenated_summed_embeddings)

        # assert that length of concatenated_summed_embeddings is 2306
        assert len(
            concatenated_summed_embeddings) == 2304, (
            f"The length of the concatenated summed embedding should be 2304, "
            f"{len(concatenated_summed_embeddings)}")

        # Create a DataFrame for the concatenated summed embedding with additional columns
        concatenated_embedding_df = pd.DataFrame([concatenated_summed_embeddings])
        concatenated_embedding_df['submitter_id'] = submitter_id
        concatenated_embedding_df['cancer'] = cancer_type

        # convert the concatenated_embedding_df to a dictionary
        concatenated_embedding_df = concatenated_embedding_df.to_dict(orient="records")[0]
        # assert that length of concatenated_embedding_df is 2306
        assert len(
            concatenated_embedding_df) == 2306, f"The length of the concatenated embedding should be 2306, {concatenated_embedding_df}"

        # Append the concatenated summed embedding DataFrame to the list
        summed_embeddings.append(concatenated_embedding_df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    summed_embeddings = pd.DataFrame(summed_embeddings)
    print(summed_embeddings)
    print("Summed embeddings created.")
    print("Saving summed embeddings...")
    # Save the summed embeddings
    summed_embeddings.to_csv(Path(save_folder, "summed_embeddings.csv"), index=False)
    print("Summed embeddings saved.")
