from argparse import ArgumentParser
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import namedtuple
from helper.load_metric_data import load_metric_data
from helper.plot_styling import color_palette, order
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_folder = Path("results", "recognizer")

Metric = namedtuple("Metric", ["name", "label"])

metrics = {
    "A": Metric("accuracy", "Accuracy"),
    "P": Metric("precision", "Precision"),
    "R": Metric("recall", "Recall"),
    "F1": Metric("f1", "F1"),
    "F1Z": Metric("f1_zeros", "F1 Zero"),
    "BA": Metric("balanced_accuracy", "Balanced Accuracy"),
    "MCC": Metric("mcc", "Matthews Correlation Coefficient")
}

model_names = {
    "multi": "DL Multi",
    "simple": "DL Simple",
    "simple_f": "DL Simple Foundation",
    "multi_f": "DL Multi Foundation",
    "baseline_m": "BL Multi",
    "baseline_s": "BL Simple"
}

# Define the dashes mapping based on style_order
dashes_dict = {
    "BL Simple": (5, 2),  # Solid line (continuous)
    "BL Multi": (5, 2),  # Solid line (continuous)
    "DL Simple": (1, 0),  # Dashed line (5px on, 2px off)
    "DL Multi": (1, 0),  # Dashed line (5px on, 2px off)
    "DL Simple Foundation": (2, 4, 2, 4, 8, 4),  # Dash-dot pattern (5px dash, 2px space, 1px dot, 2px space)
    "DL Multi Foundation": (2, 4, 2, 4, 8, 4),  # Dash-dot pattern (5px dash, 2px space, 1px dot, 2px space)
}


def create_bar_chart(metric, grouped_df: pd.DataFrame, df: pd.DataFrame, save_folder: Path):
    print(grouped_df)

    # Get unique model names dynamically
    models = df["model"].unique()

    if len(models) != 2:
        raise ValueError(f"Expected exactly two models, but found: {models}")

    # Create subplots for side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        # subset_grouped = grouped_df[grouped_df["model"] == model]
        subset_df = df[df["model"] == model]

        # Ensure all embeddings are included
        # order = subset_grouped["embedding"].unique().tolist()

        # Bar plot
        sns.barplot(x="walk_distance", y=metric.name, hue="embedding", data=subset_df,
                    palette=color_palette, hue_order=order, alpha=0.8, edgecolor="black", ax=ax)

        # Scatter plot overlay (showing all data points)
        sns.stripplot(x="walk_distance", y=metric.name, hue="embedding", data=subset_df,
                      palette=color_palette, hue_order=order, jitter=True, dodge=True, alpha=0.5, marker="o", size=6,
                      ax=ax)

        # Set title and labels
        ax.set_title(f"{metric.label} per Sample Count ({model})")
        ax.set_ylabel(metric.label)
        ax.set_xlabel("Sample Count")

        # Set y limits between 0 and 1
        ax.set_ylim(0, 1.1)

        # Remove duplicate legends from scatter plot
        if i == 1:  # Show legend only for the second subplot to avoid duplication
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(order)], labels[:len(order)], title="Embedding", loc='upper left',
                      bbox_to_anchor=(1, 1))
        else:
            ax.legend([], [], frameon=False)  # Hide legend for the first plot

    # Improve layout and save the plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_bar_chart.png"), dpi=300)
    plt.close('all')


def create_line_chart(models: List[str], metric: Metric, grouped_df: pd.DataFrame, save_folder: Path):
    # Separate the grouped data for each model.
    # Since grouped_df is multi-indexed by ["model", "walk_distance", "embedding"],
    # we extract the baseline (first model) and target (second model) data.
    baseline_df = grouped_df.loc[models[0]].reset_index()
    baseline_df["model"] = model_names[models[0]]
    baseline_model_name = model_names[models[0]]

    target_df = grouped_df.loc[models[1]].reset_index()
    target_df["model"] = model_names[models[1]]
    compare_model_name = model_names[models[1]]

    # rename model to Model and embedding to Embedding
    baseline_df.rename(columns={"model": "Model", "embedding": "Modality"}, inplace=True)
    target_df.rename(columns={"model": "Model", "embedding": "Modality"}, inplace=True)

    line_df = pd.concat([baseline_df, target_df])
    if "simple" in models and not foundation:
        y_lim = [0.5, 1.05]
    elif "simple" in models and foundation:
        y_lim = [0, 1.05]
    else:
        y_lim = [0.1, 1.05]

    plt.figure(figsize=(10, 6))

    # Plot the baseline model with reduced opacity (faded)
    sns.lineplot(
        x="walk_distance",
        y=metric.name,
        hue="Modality",
        data=line_df,
        palette=color_palette,
        hue_order=order,
        style="Model",
        style_order=[compare_model_name, baseline_model_name],
    )

    plt.title(f"{metric.label} comparison between {baseline_model_name} and {compare_model_name}")
    plt.ylabel(metric.label)
    plt.xlabel("Sample Count")
    # plt.legend(title="", loc='lower left')
    # remove legend
    plt.legend().remove()
    plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_line_plot.png"), dpi=300)
    plt.close('all')


def create_dist_line_chart(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    baseline_df = df[df["model"] == models[0]].reset_index(drop=True)
    baseline_df["model"] = model_names[models[0]]
    baseline_model_name = model_names[models[0]]

    target_df = df[df["model"] == models[1]].reset_index(drop=True)
    target_df["model"] = model_names[models[1]]
    compare_model_name = model_names[models[1]]

    # rename model to Model and embedding to Embedding
    baseline_df.rename(columns={"model": "Model", "embedding": "Modality"}, inplace=True)
    target_df.rename(columns={"model": "Model", "embedding": "Modality"}, inplace=True)

    line_df = pd.concat([baseline_df, target_df])
    line_df.reset_index(drop=True, inplace=True)
    plt.figure(figsize=(10, 6))


    # Plot the baseline model with reduced opacity (faded)
    sns.lineplot(x="walk_distance", y=metric.name, hue="Modality", data=line_df, palette=color_palette, hue_order=order,
                 style="Model", style_order=[compare_model_name, baseline_model_name],
                 dashes=dashes_dict)

    # plt.title(f"{metric.label} comparison between {baseline_model_name} and {compare_model_name}")
    plt.ylabel(metric.label)
    plt.xlabel("Sample Count")
    # plt.legend(title="", loc='lower left')
    # remove legend
    plt.legend().remove()
    plt.ylim(0.1, 1.02)
    # increase font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # set label font size
    plt.xlabel("Sample Count", fontsize=18)
    plt.ylabel(metric.label, fontsize=18)

    # remove box
    plt.box(False)
    # helvitaca font
    plt.rcParams['font.family'] = 'helvetica'
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_line_plot_comparison.png"), dpi=300)
    plt.close('all')


def create_box_plot(metric, df: pd.DataFrame, save_folder: Path):
    # Get unique model names dynamically
    models = df["model"].unique()

    if len(models) != 2:
        raise ValueError("Expected exactly two models, but found:", models)

    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        subset = df[df["model"] == model]

        # Ensure all embeddings are included
        order = subset["embedding"].unique().tolist()

        sns.boxplot(x="walk_distance", y=metric.name, hue="embedding", data=subset,
                    palette=color_palette, hue_order=order, ax=ax)

        ax.set_title(f"{metric.label} per Sample Count ({model})")
        ax.set_ylabel(metric.label)
        ax.set_xlabel("Random Selection")
        ax.legend(title="Embedding", loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_box_plot.png"), dpi=300)
    plt.close('all')


if __name__ == '__main__':
    parser = ArgumentParser(description='Aggregate metrics from recognizer results')
    parser.add_argument("-c", "--cancer", required=False, nargs='+',
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--amount_of_walk_embeddings", "-a", help="The amount of embeddings to sum", type=int,
                        required=False, default=15000)
    parser.add_argument("--models", "-m", nargs='+',
                        choices=["multi", "simple", "baseline_m", "baseline_s", "simple_f", "multi_f"],
                        default="multi",
                        help="The model to use")
    parser.add_argument("--noise_ratio", "-n", type=float, help="The noise ratio to use", default=0.0)
    parser.add_argument("--selected_metric", "-sm", required=True, choices=["A", "P", "R", "F1", "F1Z", "BA", "MCC"],
                        default="A")

    args = parser.parse_args()
    models: List[str] = args.models
    amount_of_walk_embeddings: int = args.amount_of_walk_embeddings
    cancers: List[str] = args.cancer
    selected_cancers: str = '_'.join(cancers)
    noise_ratio: float = args.noise_ratio
    selected_metric: str = args.selected_metric
    foundation: bool = "_f" in models[0] or "_f" in models[1]

    metric = metrics[selected_metric]

    if len(models) > 2:
        raise ValueError("Only two models can be compared")

    model_data = {}
    for model in models:
        logging.info(
            f"Loading data for model: {model}, cancers: {cancers}, amount_of_walk_embeddings: {amount_of_walk_embeddings},"
            f" noise_ratio: {noise_ratio}")

        if model == "baseline_m":
            model_path = "baseline/multi"
        elif model == "baseline_s":
            model_path = "baseline/simple"
        elif model == "simple_f":
            model_path = "simple"
        elif model == "multi_f":
            model_path = "multi"
        else:
            model_path = model

        model_load_folder = Path(load_folder, model_path, selected_cancers, str(amount_of_walk_embeddings))

        is_foundation = "_f" in model
        logging.info(f"Loading data from {model_load_folder}")

        # load data
        df = load_metric_data(load_folder=model_load_folder, noise_ratio=noise_ratio, foundation=is_foundation)
        df["model"] = model
        df.reset_index(drop=True, inplace=True)

        if "modality" in df.columns:
            # rename text to annotation in modality column
            df["modality"] = df["modality"].replace("Text", "Annotation")
        else:
            df["embedding"] = df["embedding"].replace("Text", "Annotation")

        if "baseline" in model:
            # rename modality to embedding
            df.rename(columns={"modality": "embedding"}, inplace=True)

            # only select up to 10 Sample Count
        df = df[df["walk_distance"] <= 10]
        if "noise" in df.columns:
            # assert only the selected noise ratio is in the df
            assert df["noise"].unique() == noise_ratio, "Noise is not unique"

        model_data[model] = df

    # combine model data
    df = pd.concat([model_data[models[0]], model_data[models[1]]])
    save_folder = Path("figures", "recognizer", selected_cancers, str(amount_of_walk_embeddings), str(noise_ratio),
                       f"{models[0]}_{models[1]}")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    logging.info(f"Saving figures to {save_folder}")

    # color palette should include only the embedding that are available in the dataset

    available_embeddings = df["embedding"].unique()
    color_palette = {k: v for k, v in color_palette.items() if k in df["embedding"].unique()}
    order = [k for k in order if k in available_embeddings]

    # create bar plot for mcc for each walk_distance and modality
    df_grouped_by_wd_embedding = df.groupby(["model", "walk_distance", "embedding"]).mean()
    df.reset_index(drop=True, inplace=True)

    annotations = df[df["embedding"] == "Annotation"]

    simple_annotation = df[(df["model"] == "simple") & (df["embedding"] == "Annotation")]
    simple_f_annotation = df[(df["model"] == "simple_f") & (df["embedding"] == "Annotation")]

    print(simple_annotation.groupby("walk_distance")["f1"].mean())
    print(simple_f_annotation.groupby("walk_distance")["f1"].mean())

    print(df)
    #create_bar_chart(metric, df_grouped_by_wd_embedding, df, save_folder)
    # create_line_chart(models,metric, df_grouped_by_wd_embedding, save_folder)
    create_dist_line_chart(models, metric, df, save_folder)
    #create_box_plot(metric, df, save_folder)
