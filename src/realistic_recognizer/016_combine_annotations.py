from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    # load all text files from data annotations
    data = []
    for file in Path("data", "annotations").glob("*.txt"):
        submitter_id = file.stem.split("_")[1]
        with open(file, "rt") as handle:
            data.append({
                "submitter_id": submitter_id,
                "text": handle.read()
            })

    # save data as dataframe
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(Path("data", "annotations", "annotations.csv"), sep=',', index=False, escapechar='\\')
