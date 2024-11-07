import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", type=str, help="Path to the input file")
parser.add_argument("--output", "-o", type=str, help="Path to the output file")
args = parser.parse_args()

file = args.file

# Read the file
df = pd.read_csv(file, sep="\t")

# Save the file
df.to_csv(args.output, index=False)
