import pandas as pd
import os, argparse
from pathlib import Path
from argparse import ArgumentParser

save_folder = Path("results", "recognizer", "aggregated_metrics")

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("--data_folder", "-d", type=Path, required=True,
                        help="Folder containing the recognizer results")
    args = parser.parse_args()

    data_folder: Path = args.data_folder

    # extract last two pieces of data_folder
    appendix = Path(*data_folder.parts[-2:])
    save_folder = Path(save_folder, appendix)

    print(f"Saving to {save_folder}")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # iterate through results recognizer folder and all its sub folders
    results = []
    split_metrics = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file == "metrics.csv":
                print("Processing", Path(root, file))
                metrics = pd.read_csv(Path(root, file))
                results.append(metrics)

            if file == 'split_metrics.csv':
                print("Processing", Path(root, file))
                split_metrics.append(pd.read_csv(Path(root, file)))

    # concatenate all metrics
    results = pd.concat(results)

    if len(split_metrics) > 0:
        split_metrics = pd.concat(split_metrics)
        split_metrics.to_csv(Path(save_folder, "split_metrics.csv"), index=False)

    # save the concatenated metrics
    results.to_csv(Path(save_folder, "metrics.csv"), index=False)
