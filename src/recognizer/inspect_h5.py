import h5py
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--file", "-f", type=Path, required=True)

args = parser.parse_args()

file = args.file
keys = ['Image', 'Mutation', 'RNA', 'Text']

with h5py.File(file) as h5_file:
    print(h5_file.keys())
    for key in h5_file.keys():
        print(f"Key: {key}")
        print(f"Shape: {h5_file[key].shape}")
        print(f"Type: {h5_file[key].dtype}")
        print(f"Value: {h5_file[key][()]}")

    # create df from h5 file using the keys and put them togheter in a df

    df = pd.DataFrame({key: h5_file[key][()] for key in keys})
    df["Total"] = df["Image"] + df["Mutation"] + df["RNA"] + df["Text"]
    print(df.head())