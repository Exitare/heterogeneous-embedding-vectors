import pandas as pd
from pathlib import Path
from tqdm import tqdm

save_folder = Path("results", "classifier", "mappings")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load data
    brca_data = pd.read_csv(Path("data", "rna", "BRCA", "data.csv"), index_col=0)
    brca_data["Cancer"] = "BRCA"
    laml_data = pd.read_csv(Path("data", "rna", "LAML", "data.csv"), index_col=0)
    laml_data["Cancer"] = "LAML"
    coad_data = pd.read_csv(Path("data", "rna", "COAD", "data.csv"), index_col=0)
    coad_data["Cancer"] = "COAD"
    blca_data = pd.read_csv(Path("data", "rna", "BLCA", "data.csv"), index_col=0)
    blca_data["Cancer"] = "BLCA"
    stad_data = pd.read_csv(Path("data", "rna", "STAD", "data.csv"), index_col=0)
    stad_data["Cancer"] = "STAD"
    thca_data = pd.read_csv(Path("data", "rna", "THCA", "data.csv"), index_col=0)
    thca_data["Cancer"] = "THCA"

    # Concatenate data
    tcga_data = pd.concat([brca_data, laml_data, coad_data, blca_data, stad_data, thca_data], axis=0)

    # Load data json
    manifest_file = pd.read_json(Path("data", "annotations", "manifest.json"))

    # extract all the cases from the manifest file
    cases = manifest_file["cases"]
    # get all submitter ids from the cases
    annotations_submitter_ids = [case[0]["submitter_id"] for case in cases]

    mutations = pd.read_csv(Path("data", "mutations", "mutations.csv"))

    mappings: list = []
    # Search for overlap between patient and submitter_id
    for patient_case in tcga_data["Patient"]:
        found_annotations = [annotation for annotation in annotations_submitter_ids if
                             annotation in patient_case]

        # find patients in mutations
        found_mutations = [mutation for mutation in mutations["submitter_id"].values if patient_case in mutation]

        # Always include the patient and cancer information
        base_row = {
            "patient": patient_case,
            "cancer": tcga_data[tcga_data["Patient"] == patient_case]["Cancer"].values[0]
        }

        if found_annotations:
            for annotation in found_annotations:
                row = base_row.copy()  # Create a copy of the base row for each annotation
                row["submitter_id"] = annotation

                if found_mutations:
                    for mutation in found_mutations:
                        mutation_row = row.copy()  # Create a copy for each mutation
                        mutation_row["mutation_id"] = mutation
                        mappings.append(mutation_row)  # Add the row to the list
                else:
                    mappings.append(row)  # Add the row without mutations if no mutations are found
        else:
            # If no annotations, just handle mutations or add the base row
            if found_mutations:
                for mutation in found_mutations:
                    row = base_row.copy()
                    row["mutation_id"] = mutation
                    mappings.append(row)
            else:
                mappings.append(base_row)  # Add the base row if no annotations or mutations are found

    # Convert mappings to DataFrame
    mappings_df = pd.DataFrame(mappings)
    # only keep distinct mappings
    mappings_df.drop_duplicates(inplace=True)
    print(mappings_df)

    # Save the mappings to a CSV file
    mappings_df.to_csv(Path(save_folder, "mappings.csv"), index=False)
    print(f"Saved {len(mappings)} mappings.")
