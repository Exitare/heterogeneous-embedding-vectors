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
    mutations = pd.read_csv(Path("data", "mutations", "mutations.csv"))

    mappings: list = []
    # Search for overlap between patient and submitter_id
    for case in manifest_file["cases"]:
        case = case[0]
        submitter_id = case["submitter_id"]
        # Check if case submitter id is in the Patient column of tcga_data
        # Check that submitter id is part of the tcga data patient value as a substring
        found_patients = [patient for patient in tcga_data["Patient"].values if submitter_id in patient]
        # find patients in mutations
        found_mutations = [patient for patient in mutations["submitter_id"].values if submitter_id in patient]

        row = {}
        if found_patients:
            for patient in found_patients:
                row["submitter_id"] = submitter_id
                row["patient"] = patient
                row["cancer"] = tcga_data[tcga_data["Patient"] == patient]["Cancer"].values[0]

        if found_mutations:
            for patient in found_mutations:
                row["mutation_id"] = patient

        mappings.append(row)

    # Convert mappings to DataFrame
    mappings_df = pd.DataFrame(mappings)
    # only keep distinct mappings
    mappings_df.drop_duplicates(inplace=True)
    # only keep mappings that are available in both files
    mappings_df = mappings_df[mappings_df["patient"].isin(tcga_data["Patient"].values)]

    # Save the mappings to a CSV file
    mappings_df.to_csv(Path(save_folder, "realistic_mappings.csv"), index=False)
    print(f"Saved {len(mappings)} mappings.")
