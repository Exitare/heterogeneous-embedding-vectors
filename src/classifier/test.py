import sys

import pandas as pd
from pathlib import Path

# v1 = pd.read_csv(Path("data", "bmeg", "LAML", "data.csv"))
# print(v1.shape)
# input()
# v2 = pd.read_csv(Path("data", "bmeg", "v_2", "LAML", "data.csv"), index_col=0)
# reset index
# v2.reset_index(drop=False, inplace=True)
# rename index column to patient
# v2.rename(columns={"index": "Patient"}, inplace=True)
# print(v2.shape)
# input()

# assert v1.shape == v2.shape, "Dataframes should have the same shape"

# ['TCGA-CE-A27D', 'TCGA-BH-A0HF', 'TCGA-G2-A3IB', 'TCGA-D8-A1JN',
#        'TCGA-AC-A2FB', 'TCGA-A6-5659', 'TCGA-CF-A3MI', 'TCGA-EL-A3H7',
#        'TCGA-EM-A2P0', 'TCGA-H2-A421']
# find file that contains TCGA-CE-A27D in data annotations
for file in Path("data", "annotations").glob("*.txt"):
    if "TCGA-CD-8524" in file.stem:
        print(file.stem)
        with open(file, "rt") as handle:
            data = handle.read()
            print("File found")
            break
else:
    print("File not found")

# print the amunt of .s in the file
print(data.count("."))
print(data)
