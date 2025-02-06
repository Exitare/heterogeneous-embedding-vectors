import h5py
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import sys

parser = ArgumentParser()
parser.add_argument("--file", "-f", type=Path, required=True)

args = parser.parse_args()

file = args.file
keys = ['Image', 'Mutation', 'RNA', 'Text']

with h5py.File(file) as h5_file:
    print(h5_file.keys())
    for key in h5_file.keys():
        print(f"Key: {key}")
        if key == "meta_information":
            continue
        print(f"Shape: {h5_file[key].shape}")
        print(f"Type: {h5_file[key].dtype}")
        print(f"Value: {h5_file[key][()]}")

    # create df from h5 file using the keys and put them togheter in a df

    df = pd.DataFrame({key: h5_file[key][()] for key in keys})
    df["Total"] = df["Image"] + df["Mutation"] + df["RNA"] + df["Text"]
    # find all unique combintaions of the columns
    unique_combinations = df.drop_duplicates()
    # print(f"Unique combinations: {unique_combinations}")
    # check if there

    # Define a relative threshold (e.g., 50% of the total)
    relative_threshold = 0.75


    # Identify oversampled features
    def find_oversampled_features_relative(row, relative_threshold):
        oversampled = [
            feature
            for feature, count in row.items()
            if feature != "Total" and count / row["Total"] > relative_threshold
        ]
        return oversampled


    df["Oversampled"] = df.apply(lambda row: find_oversampled_features_relative(row, relative_threshold), axis=1)
    print(df)
    # calculate the percentage of rows where oversampling happened
    percentage_oversampling = len(df[df["Oversampled"].apply(len) > 0]) / len(df)
    # print(f"Percentage of oversampling: {percentage_oversampling:.2f}")

keys = ['Text', 'THCA', 'BRCA', 'LUAD', 'COAD', 'STAD', 'BLCA', 'RNA', 'Image', 'Mutation']
df = pd.DataFrame()
with h5py.File(file) as h5_file:
    for key in h5_file.keys():
        # add keys Text, THCA, BRCA, LUAD, COAD, STAD, BLCA, RNA, Image, Mutation
        if key in keys:
            df[key] = h5_file[key][()]


with h5py.File(file) as h5_file:
    x = h5_file["X"]
    print(f"X: {x[:]}")

df["Total"] = df["Image"] + df["Mutation"] + df["RNA"] + df["Text"]
# Set Pandas to display all columns permanently
pd.set_option('display.max_columns', None)
# show all columns of dataset
print(df)



cancers = ['THCA', 'BRCA', 'LUAD', 'COAD', 'STAD', 'BLCA']


if not set(cancers).issubset(df.columns):
    sys.exit("Not all cancer types are present in the dataset.")


# print all unique total amounts
print(f"Unique total amounts: {df['Total'].unique()}")

# Validate unique cancer condition
violations = df[cancers].apply(lambda row: (row > 0).sum() > 1, axis=1)

# Output results
if violations.any():
    print("Violations found! These rows do not satisfy the condition:")
    print(df[violations])
else:
    print("No violations. The condition is satisfied.")


# ✅ Ensure each cancer type has at least one non-zero row
missing_cancers = [cancer for cancer in cancers if (df[cancer] > 0).sum() == 0]

if missing_cancers:
    print(f"❌ Error: The following cancers have no occurrences in the dataset: {', '.join(missing_cancers)}")
else:
    print("✅ All cancer types have at least one occurrence in the dataset.")

if len(df['Total'].unique()) > 1:
    for total in df['Total'].unique():
        missing_cancers = [cancer for cancer in cancers if (df[cancer] > 0).sum() == 0 and df['Total'] == total]
        if missing_cancers:
            print(f"❌ Error: The following cancers have no occurrences in the dataset: {', '.join(missing_cancers)}")
        else:
            print("✅ All cancer types have at least one occurrence in the dataset.")