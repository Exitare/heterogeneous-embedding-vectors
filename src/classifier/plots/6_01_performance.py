from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from click import style

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

            print(iteration)
            try:
                df = pd.read_csv(Path(iteration, "results.csv"))
                # load the results from the run
                results.append(df)
            except FileNotFoundError:
                continue

    # concatenate all results
    results = pd.concat(results)
    # create new column walks by combining walk distance and amount_of_walk using an underscore
    results["walks"] = results["walk_distance"].astype(str) + "_" + results["amount_of_walks"].astype(str)
    print(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="cancer", y="accuracy", hue="cancer", ax=ax)
    ax.set_title("Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"accuracy.png"), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="cancer", y="f1", hue="cancer", ax=ax)
    ax.set_title("F1 score")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"f1_score.png"), dpi=300)
    plt.close('all')


    # create a grid plot for the accuracy, each unique value of walks should have a separate plot, the hue is cancer
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="walks", y="accuracy", hue="cancer", ax=ax)
    ax.set_title("Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Walks")
    # adjust legend
    ax.legend(loc='lower center', ncols=7, title="Cancer Type", bbox_to_anchor=(0.5, -0.28))
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"accuracy_grid.png"), dpi=300)



