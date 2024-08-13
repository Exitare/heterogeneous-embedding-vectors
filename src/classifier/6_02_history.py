from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

save_folder = Path("figures", "classifier")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    # load all runs from results/classifier/classification
    results = []
    # iterate over all subfolders
    cancer_folder = Path("results", "classifier", "classification", cancers)
    for run in cancer_folder.iterdir():
        if run.is_file():
            continue

        print(run)
        try:
            df = pd.read_csv(Path(run, "history.csv"))
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
    #print(results)
    # calculate the mean
    mean_results = results.groupby("epoch").mean().reset_index()
    print(mean_results)


    # plot mean results for all training epochs
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.lineplot(data=mean_results, x="epoch", y="loss", label="Loss")
    sns.lineplot(data=mean_results, x="epoch", y="accuracy", label="Accuracy")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"history.png"), dpi=300)


