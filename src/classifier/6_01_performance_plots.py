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
    print("Selected cancers: ", selected_cancers)
    cancers = "_".join(selected_cancers)

    # load all runs from results/classifier/classification
    results = []
    # iterate over all subfolders
    cancer_folder = Path("results", "classifier", "classification", cancers)
    for run in cancer_folder.iterdir():
        if not run.is_file():
            continue

        # if file is not called results.csv continue
        if run.name != "results.csv":
            continue

        # load the results from the run
        results.append(pd.read_csv(run))

    # concatenate all results
    results = pd.concat(results)
    print(results)

    # create boxplot for performance

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, y="accuracy", ax=ax)
    ax.set_title("Accuracy of classifier")
    ax.set_ylabel("Accuracy")
    plt.savefig(Path(save_folder, f"{cancers}_accuracy.png"), dpi=300)
