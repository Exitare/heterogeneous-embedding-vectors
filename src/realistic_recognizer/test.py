import pandas as pd
from pathlib import Path

v1 = pd.read_csv(Path("data", "bmeg", "LAML", "data.csv"))
#print(v1.shape)
#input()
v2 = pd.read_csv(Path("data", "bmeg", "v_2", "LAML", "data.csv"), index_col=0)
# reset index
#v2.reset_index(drop=False, inplace=True)
# rename index column to patient
#v2.rename(columns={"index": "Patient"}, inplace=True)
#print(v2.shape)
#input()

assert v1.shape == v2.shape, "Dataframes should have the same shape"