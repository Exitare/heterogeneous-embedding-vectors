from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load folder
load_folder = Path("results", "recognizer")
save_folder = Path("figures", "recognizer")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # load history
    history = pd.read_csv(Path(load_folder, "history.csv"))
    # reset index
    history = history.reset_index()
    # rename to epoch
    history = history.rename(columns={"index": "epoch"})

    # plot history
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.lineplot(data=history, x="epoch", y="loss", label="loss")
    sns.lineplot(data=history, x="epoch", y="val_loss", label="val_loss")
    sns.lineplot(data=history, x="epoch", y="val_output_0_mae", label="val_loss_text")
    sns.lineplot(data=history, x="epoch", y="val_output_1_mae", label="val_loss_image")
    sns.lineplot(data=history, x="epoch", y="val_output_2_mae", label="val_mae_rna")
    plt.savefig(Path(save_folder, "history.png"), dpi=300)
