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
            df = pd.read_csv(Path(run, "results.csv"))
            # load the results from the run
            results.append(df)
        except FileNotFoundError:
            continue

    # concatenate all results
    results = pd.concat(results)
    print(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="cancer", y="accuracy", hue="cancer", ax=ax)
    ax.set_title("Accuracy of classifier")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"accuracy.png"), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="cancer", y="f1", hue="cancer", ax=ax)
    ax.set_title("Accuracy of classifier")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"f1_score.png"), dpi=300)
    plt.close('all')


