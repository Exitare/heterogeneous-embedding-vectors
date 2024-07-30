import pandas as pd
from pathlib import Path
from tqdm import tqdm

save_folder = Path("results", "mappings")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load data
    brca_data = pd.read_csv(Path("data", "bmeg", "v_2", "BRCA", "data.csv"), index_col=0)
    laml_data = pd.read_csv(Path("data", "bmeg", "v_2", "LAML", "data.csv"), index_col=0)
    coad_data = pd.read_csv(Path("data", "bmeg", "v_2", "COAD", "data.csv"), index_col=0)
    blca_data = pd.read_csv(Path("data", "bmeg", "v_2", "BLCA", "data.csv"), index_col=0)
    stad_data = pd.read_csv(Path("data", "bmeg", "v_2", "STAD", "data.csv"), index_col=0)
    thca_data = pd.read_csv(Path("data", "bmeg", "v_2", "THCA", "data.csv"), index_col=0)

    # Concatenate data
    tcga_data = pd.concat([brca_data, laml_data, coad_data, blca_data, stad_data, thca_data], axis=0)

    # Load data json
    manifest_file = pd.read_json(Path("data", "annotations", "manifest.json"))

    mappings: list = []
    # Search for overlap between patient and submitter_id
    for case in manifest_file["cases"]:
        case = case[0]
        submitter_id = case["submitter_id"]
        # Check if case submitter id is in the Patient column of tcga_data
        # Check that submitter id is part of the tcga data patient value as a substring
        found_patients = [patient for patient in tcga_data["Patient"].values if submitter_id in patient]

        if found_patients:
            for patient in found_patients:
                mappings.append({
                    "submitter_id": submitter_id,
                    "patient": patient
                })

    # Convert mappings to DataFrame
    mappings_df = pd.DataFrame(mappings)
    # only keep distinct mappings
    mappings_df.drop_duplicates(inplace=True)
    # only keep mappings that are available in both files
    mappings_df = mappings_df[mappings_df["patient"].isin(tcga_data["Patient"].values)]

    # Save the mappings to a CSV file
    mappings_df.to_csv(Path(save_folder, "realistic_mappings.csv"), index=False)
    print(f"Saved {len(mappings)} mappings.")
