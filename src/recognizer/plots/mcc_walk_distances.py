from argparse import ArgumentParser
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_folder = Path("results", "recognizer")

if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--model", "-m", choices=["multi", "simple", "baseline"], default="multi",
                        help="The model to use")
    parser.add_argument("--foundation", "-f", action="store_true", help="Use of the foundation model metrics")

    args = parser.parse_args()
    model: str = args.model
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: [str] = args.cancer
    foundation: bool = args.foundation
    selected_cancers: [str] = '_'.join(cancers)

    logging.info(
        f"Loading data for model: {model}, cancers: {cancers}, foundation: {foundation}, amount_of_walk_embeddings: {amount_of_walk_embeddings}")

    load_folder = Path(load_folder, model, selected_cancers, str(amount_of_walk_embeddings))

    dfs = []
    for noise_folder in load_folder.iterdir():
        if noise_folder.is_file():
            continue

        for walk_distance_folder in noise_folder.iterdir():
            if walk_distance_folder.is_file():
                continue

            if 'combined_embeddings' in walk_distance_folder.parts and not foundation:
                continue

            if 'combined_embeddings' not in walk_distance_folder.parts and foundation:
                logging.info(f"Skipping {walk_distance_folder} because foundation is set to True")
                continue

            for run_folder in walk_distance_folder.iterdir():
                if run_folder.is_file():
                    continue

                for file in run_folder.iterdir():
                    if 'combined_embeddings' in walk_distance_folder.parts:
                        file_name = "split_metrics.csv"
                    else:
                        file_name = "metrics.csv"

                    if file.is_file() and file_name in file.parts:
                        df = pd.read_csv(file)
                        dfs.append(df)

    df = pd.concat(dfs)
    print(df)

    print(df["mcc"].isna().sum())

    #create bar plot for mcc for each walk_distance and modality
    df = df.groupby(["walk_distance", "em"]).mean()
    fig, ax = sns.barplot(x="walk_distance", y="mcc", hue="modality", data=df)
    fig.set_title("MCC per walk distance and modality")
    fig.set_ylabel("MCC")

    fig.tight_layout()
    fig.show()
