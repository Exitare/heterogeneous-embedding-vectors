from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

save_folder = Path("figures", "classifier")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, "performance")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load all runs from results/classifier/classification
    results = []
    # iterate over all subfolders
    cancer_folder = Path("results", "classifier", "classification", cancers)
    for run in cancer_folder.iterdir():
        if run.is_file():
            continue

        for iteration in run.iterdir():
            if iteration.is_file():
                continue

            walk_distance: int = iteration.parts[-2].split("_")[0]
            walk_amount: int = iteration.parts[-2].split("_")[1]
            try:
                df = pd.read_csv(Path(iteration, "history.csv"))
                # add walk distance and amount of walks
                df["walk_distance"] = int(walk_distance)
                df["walk_amount"] = int(walk_amount)
                # reset index
                df = df.reset_index()
                # rename to epoch
                df = df.rename(columns={"index": "epoch"})
                # load the results from the run
                results.append(df)
            except FileNotFoundError:
                continue

    # concatenate all results
    results = pd.concat(results)

    # calculate the mean
    mean_results = results.groupby(["epoch", "walk_distance", "walk_amount"]).mean().reset_index()

    # plot mean results for all training epochs
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.lineplot(data=mean_results, x="epoch", y="loss", label="Loss")
    sns.lineplot(data=mean_results, x="epoch", y="accuracy", label="Accuracy")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"history.png"), dpi=300)
