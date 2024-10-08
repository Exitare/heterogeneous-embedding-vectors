import pandas as pd
from pathlib import Path

cancers = ["BLCA", "BRCA", "LUAD", "THCA", "COAD", "STAD"]

if __name__ == '__main__':

    for cancer in cancers:
        rna = pd.read_csv(Path("data", "rna", cancer, "data.csv"), index_col=0)

        print(f"Amount of patients in {cancer}: {rna['Patient'].nunique()}")
