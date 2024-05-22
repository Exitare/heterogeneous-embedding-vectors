import pandas as pd
import os, argparse
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("--data_folder", "-d", type=Path, required=True,
                        help="Folder containing the recognizer results")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Folder to save the aggregated metrics")
    args = parser.parse_args()

    data_folder: Path = args.data_folder
    save_path: Path = args.output

    if not save_path.exists():
        save_path.mkdir(parents=True)

    # iterate through results recognizer folder and all its sub folders
    results = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file == "metrics.csv":
                print("Processing", Path(root, file))
                metrics = pd.read_csv(Path(root, file))
                results.append(metrics)

    # concatenate all metrics
    results = pd.concat(results)

    # save the concatenated metrics
    results.to_csv(Path(save_path, "metrics.csv"), index=False)
