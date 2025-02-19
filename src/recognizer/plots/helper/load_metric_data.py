from pathlib import Path
import logging
import pandas as pd

def load_metric_data(load_folder: Path, noise_ratio: float, foundation: bool, verbose: bool = False) -> pd.DataFrame:
    dfs = []
    for noise_folder in load_folder.iterdir():
        if noise_folder.is_file():
            logging.info(f"Skipping {noise_folder} because it is a file")
            continue

        if str(noise_ratio) not in noise_folder.parts and noise_ratio != -1:
            logging.info(f"Skipping {noise_folder} because noise_ratio is not -1")
            continue

        for walk_distance_folder in noise_folder.iterdir():
            if walk_distance_folder.is_file():
                logging.info(f"Skipping {walk_distance_folder} because it is a file")
                continue


            if 'combined_embeddings' in walk_distance_folder.parts and not foundation:
                continue

            if 'combined_embeddings' not in walk_distance_folder.parts and foundation:
                logging.info(f"Skipping {walk_distance_folder} because foundation is set to True")
                continue

            for run_folder in walk_distance_folder.iterdir():
                if run_folder.is_file():
                    logging.info(f"Skipping {run_folder} because it is a file")
                    continue

                for file in run_folder.iterdir():
                    file_name = "metrics.csv"
                    if file.is_file() and file_name in file.parts:
                        if verbose:
                            logging.info(f"Loading {file}...")
                        df = pd.read_csv(file)
                        dfs.append(df)

    return pd.concat(dfs)