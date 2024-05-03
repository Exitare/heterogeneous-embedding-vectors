import pandas as pd
import os, argparse
from pathlib import Path

save_path = Path("results", "aggregated_metrics")

if __name__ == '__main__':
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # iterate through results recognizer folder and all its sub folders
    results_path = Path("results", "recognizer")
    results = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file == "metrics.csv":
                print("Processing", Path(root, file))
                metrics = pd.read_csv(Path(root, file))
                results.append(metrics)

    # concatenate all metrics
    results = pd.concat(results)

    # save the concatenated metrics
    results.to_csv(Path(save_path, "metrics.csv"), index=False)
