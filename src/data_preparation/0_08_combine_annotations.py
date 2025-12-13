from pathlib import Path
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    print("Selected cancers: ", selected_cancers)

    cancers = "_".join(selected_cancers)

    save_folder = Path("data", "annotations", cancers)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load all text files from data annotations
    data = []

    for cancer in selected_cancers:
        for file in Path("data", "annotations", f"TCGA-{cancer}").glob("*.txt"):
            submitter_id = file.stem.split("_")[1]
            with open(file, "rt") as handle:
                data.append({
                    "submitter_id": submitter_id,
                    "text": handle.read(),
                    "cancer": cancer
                })

    # save data as dataframe
    df = pd.DataFrame(data)
    print(df.head())
    # assert that all cancers are in the cancer column
    assert all(
        [cancer in df["cancer"].unique() for cancer in selected_cancers]), "All cancers should be in the cancer column"
    df.to_csv(Path(save_folder, "annotations.csv"), sep=',', index=False, escapechar='\\')
