import pandas as pd
from pathlib import Path
import sys

load_folder = Path("data", "annotations")

if __name__ == '__main__':
    empty_txts = []

    annotations = pd.read_csv(Path(load_folder, "annotations.csv"))

    for submitter_id in annotations["submitter_id"]:

        # split the submitter text using the .
        text = annotations[annotations["submitter_id"] == submitter_id]["text"].values[0]

        # check if text only consists of whitespace
        if text.isspace():
            empty_txts.append({
                "submitter_id": submitter_id,
            })

    if not empty_txts:
        print("No empty texts found.")
        sys.exit(0)

    print(f"Number of empty texts: {len(empty_txts)}")
    print(empty_txts)
    empty_txts = pd.DataFrame(empty_txts)
    empty_txts.to_csv(Path(load_folder, "empty_txts.csv"), index=False)
