from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statannotations.Annotator import Annotator

save_folder = Path("figures", "classifier_holdout")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "Matthews Correlation Coefficient",
    "balanced_accuracy": "Balanced Accuracy",
}

color_palette = {
    "Annotation": "#c8b7b7ff",
    "Image": "#d38d5fff",
    "RNA": "#c6afe9ff",
    "Mutation": "#de87aaff",
    "BRCA": "#c837abff",
    "LUAD": "#37abc8ff",
    "BLCA": "#ffcc00ff",
    "THCA": "#d35f5fff",
    "STAD": "#f47e44d7",
    "COAD": "#502d16ff",
    "All": "#000000",
    "3_3": "#c837abff",
    "3_4": "#ffcc00ff",
    "3_5": "#37abc8ff",
    "4_3": "#e63946ff",
    "4_4": "#2a9d8fff",
    "4_5": "#f4a261ff",
    "5_3": "#264653ff",
    "5_4": "#8ecae6ff",
    "5_5": "#ff6700ff",
    "6_6": "#6a0572ff"
}


def _build_pairs_within_x(df: pd.DataFrame, x_col: str, hue_col: str, hue_order: list[str]) -> list[
    tuple[tuple, tuple]]:
    pairs: list[tuple[tuple, tuple]] = []
    if not len(hue_order):
        return pairs
    baseline = hue_order[0]
    x_levels = sorted([x for x in df[x_col].dropna().unique().tolist()])
    for xv in x_levels:
        for h in hue_order:
            if h == baseline:
                continue
            pairs.append(((xv, baseline), (xv, h)))
    return pairs


def _apply_stats_annotations(ax, df: pd.DataFrame, x: str, y: str, hue: str, pairs: list[tuple[tuple, tuple]],
                             hue_order: list[str]):
    annot = Annotator(
        ax=ax,
        pairs=pairs,
        data=df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order
    )
    annot.configure(
        test="Mann-Whitney",
        comparisons_correction="Benjamini-Hochberg",
        loc="inside",
        text_format="star",
        verbose=1,
        show_test_name=False
    )
    annot.apply_and_annotate()


def create_grid_plot(df: pd.DataFrame, metric: str):
    hue_order = ["3_3", "3_4", "3_5", "4_3", "4_4", "4_5", "5_3", "5_4", "5_5"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxenplot(
        data=df, x="cancer", y=metric, hue="walks",
        ax=ax, palette=color_palette, hue_order=hue_order
    )

    ax.set_title(f"{metrics.get(metric, metric)} Score")
    ax.set_ylabel(f"{metrics.get(metric, metric)} Score")
    ax.set_xlabel("")
    if df[metric].notna().any():
        top99 = float(df[metric].quantile(0.99))
        ax.set_ylim(0.0, min(1.10, max(0.95, top99) + 0.06))
    else:
        ax.set_ylim(0.0, 1.0)

    handles, labels = ax.get_legend_handles_labels()
    label_to_pretty = {w: f"SC: {w.split('_')[0]}  R: {w.split('_')[1]}" for w in hue_order}
    new_labels = [label_to_pretty.get(l, l) for l in labels]
    ax.legend(handles, new_labels, loc='lower center', ncols=5, title="Combination",
              bbox_to_anchor=(0.5, -0.28), frameon=False)

    pairs = _build_pairs_within_x(df, x_col="cancer", hue_col="walks", hue_order=hue_order)
    _apply_stats_annotations(ax, df, x="cancer", y=metric, hue="walks", pairs=pairs, hue_order=hue_order)

    yticks = [yt for yt in ax.get_yticks() if yt <= 1.0]
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{yt:.2f}" for yt in yticks])

    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score_grid.png"), dpi=300)
    plt.close('all')


def create_selected_walks_boxenplot(df: pd.DataFrame, metric: str, wanted: list[str] = None):
    """
    Boxenplot for walks 3_3, 4_4, 5_5 only, with stats (baseline = 3_3) per cancer.
    Saves to: <metric>_selected_walks_3_3_4_4_5_5_box_annotated.png
    """
    if wanted is None:
        wanted = ["3_3", "4_4", "5_5"]
    sub = df[df["walks"].isin(wanted)].copy()
    if sub.empty or metric not in sub.columns:
        logging.warning(f"No data for selected walks or missing metric '{metric}'.")
        return

    # keep per-cancer view; drop "All" aggregate if present
    sub = sub[sub["cancer"] != "All"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxenplot(
        data=sub, x="cancer", y=metric, hue="walks",
        ax=ax, palette=color_palette, hue_order=wanted
    )

    ax.set_title(f"{metrics.get(metric, metric)} {' '.join(modalities.split('_')).capitalize()}")
    ax.set_ylabel(f"{metrics.get(metric, metric)}")
    ax.set_xlabel("")

    if sub[metric].notna().any():
        top99 = float(sub[metric].quantile(0.99))
        ax.set_ylim(0.45, min(1.10, max(0.95, top99) + 0.06))

    handles, labels = ax.get_legend_handles_labels()
    label_to_pretty = {w: f"SC: {w.split('_')[0]}  R: {w.split('_')[1]}" for w in wanted}
    new_labels = [label_to_pretty.get(l, l) for l in labels]
    ax.legend(handles, new_labels, loc='lower center', ncols=3, title="Combination",
              bbox_to_anchor=(0.5, -0.28), frameon=False)

    pairs = _build_pairs_within_x(sub, x_col="cancer", hue_col="walks", hue_order=wanted)
    _apply_stats_annotations(ax, sub, x="cancer", y=metric, hue="walks", pairs=pairs, hue_order=wanted)

    yticks = [yt for yt in ax.get_yticks() if yt <= 1.0]
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{yt:.2f}" for yt in yticks])

    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_selected_walks_{'_'.join(wanted)}_box_annotated.png"), dpi=300)
    plt.close('all')


def create_hue_performance_plot(df: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="cancer", y=metric, hue="cancer", ax=ax, palette=color_palette)
    ax.set_title(f"{metrics.get(metric, metric)} score")
    ax.set_ylabel(f"{metrics.get(metric, metric)} score")
    ax.set_xlabel("Cancer")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_score.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot(df: pd.DataFrame):
    sub_df = df[df["cancer"] == "All"].copy()
    sub_df = sub_df.melt(
        id_vars=["cancer"],
        value_vars=["precision", "recall", "auc"],
        var_name="metric",
        value_name="score"
    )
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC"}
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub_df, x="cancer", y="score", hue="metric", ax=ax)
    ax.set_title("Overview scores")
    ax.set_ylabel("Overview scores")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(Path(save_folder, "overview_scores.png"), dpi=300)
    plt.close('all')


def create_performance_overview_plot_per_combination(df: pd.DataFrame):
    sub_df = df[df["cancer"] == "All"].copy()
    sub_df = sub_df.melt(
        id_vars=["walks"],
        value_vars=["precision", "recall", "auc", "f1", "mcc", "balanced_accuracy"],
        var_name="metric",
        value_name="score"
    )
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC", "accuracy": "Accuracy",
         "f1": "F1 Score", "mcc": "MCC", "balanced_accuracy": "Balanced Accuracy"}
    )
    sub_df["walks"] = sub_df["walks"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sub_df, x="walks", y="score", hue="metric", ax=ax)
    ax.set_title("Metrics")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim(0.8, 1.0)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [f"Distance: {label.split(' ')[0]}\n Amount: {label.split(' ')[1]}" for label in labels]
    ax.set_xticklabels(labels)
    plt.legend(title="", loc='lower right', ncols=7, frameon=False)
    plt.tight_layout()
    plt.savefig(Path(save_folder, "overview_scores_by_walks.png"), dpi=300)
    plt.close('all')


def create_performance_overview_heatmap(df: pd.DataFrame):
    sub_df = df[df["cancer"] == "All"].copy()
    sub_df = sub_df.melt(
        id_vars=["walk_distance", "amount_of_walks"],
        value_vars=["precision", "recall", "auc", "accuracy", "f1", "mcc", "balanced_accuracy"],
        var_name="metric",
        value_name="score"
    )
    sub_df["metric"] = sub_df["metric"].replace(
        {"precision": "Precision", "recall": "Recall", "auc": "AUC", "accuracy": "Accuracy",
         "f1": "F1 Score", "mcc": "MCC", "balanced_accuracy": "Balanced Accuracy"}
    )
    for metric in ["Precision", "Recall", "AUC", "Accuracy", "F1 Score", "MCC", "Balanced Accuracy"]:
        metric_data = sub_df[sub_df["metric"] == metric].pivot_table(
            index="walk_distance", columns="amount_of_walks", values="score", aggfunc="mean"
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(metric_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5, vmin=0.83, vmax=0.92)
        plt.title(f"{metric} Scores Heatmap")
        plt.xlabel("Sample Counts")
        plt.ylabel("Repeats")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Path(save_folder, f"overview_scores_heatmap_{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def create_overall_walks_boxplot_with_stats(df: pd.DataFrame, metric: str):
    if "walks" not in df.columns or metric not in df.columns:
        logging.warning(f"Cannot build overall-by-walks stats for metric '{metric}'.")
        return
    sub = df[df["cancer"] != "All"].copy()
    hue_order = ["3_3", "3_4", "3_5", "4_3", "4_4", "4_5", "5_3", "5_4", "5_5"]
    sub = sub[sub["walks"].isin(hue_order)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=sub, x="walks", y=metric, order=hue_order, palette=color_palette, ax=ax)
    ax.set_title(f"{metrics.get(metric, metric)} by Walks (across all cancers)")
    ax.set_xlabel("Walks (SC_R)")
    ax.set_ylabel(metrics.get(metric, metric))
    pairs = [((hue_order[0]), h) for h in hue_order[1:]]
    annot = Annotator(ax=ax, pairs=pairs, data=sub, x="walks", y=metric, order=hue_order)
    annot.configure(
        test="Mann-Whitney",
        comparisons_correction="Benjamini-Hochberg",
        loc="inside",
        text_format="star",
        verbose=1,
        show_test_name=False
    )
    annot.apply_and_annotate()
    if sub[metric].notna().any():
        top99 = float(sub[metric].quantile(0.99))
        ax.set_ylim(0.0, min(1.10, max(0.95, top99) + 0.06))
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric}_overall_by_walks_box_annotated.png"), dpi=300)
    plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="The cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--modalities", "-m", nargs="+", default=["annotations", "images", "mutations", "rna"],
                        choices=["rna", "annotations", "mutations", "images"])
    args = parser.parse_args()

    selected_cancers = args.cancer
    selected_modalities: list[str] = args.modalities

    logging.info(f"Using cancer types: {selected_cancers}")
    logging.info(f"Using modalities: {selected_modalities}")

    cancers = "_".join(selected_cancers)
    modalities = '_'.join(selected_modalities)

    save_folder = Path(save_folder, cancers, modalities, "performance")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    results = []
    cancer_folder = Path("results", "classifier_holdout", "classification", cancers, modalities)
    for run in cancer_folder.iterdir():
        if run.is_file():
            continue
        for iteration in run.iterdir():
            if iteration.is_file():
                continue
            logging.info(iteration)
            try:
                df = pd.read_csv(Path(iteration, "results.csv"))
                results.append(df)
            except FileNotFoundError:
                continue

    if not results:
        raise FileNotFoundError(f"No results.csv found under {cancer_folder}/**/")

    results = pd.concat(results, ignore_index=True)

    if "walk_distance" in results.columns and "amount_of_walks" in results.columns:
        results["walks"] = results["walk_distance"].astype(str) + "_" + results["amount_of_walks"].astype(str)
    else:
        logging.warning("walk_distance and/or amount_of_walks missing — 'walks' column will not be created.")
        results["walks"] = results.get("walks", pd.Series(dtype=str))

    results.reset_index(drop=True, inplace=True)

    create_hue_performance_plot(results, "accuracy")
    create_hue_performance_plot(results, "f1")
    create_performance_overview_plot(results)
    create_performance_overview_plot_per_combination(results)
    create_performance_overview_heatmap(results)

    create_grid_plot(results, "accuracy")
    create_grid_plot(results, "f1")
    create_overall_walks_boxplot_with_stats(results, "accuracy")
    create_overall_walks_boxplot_with_stats(results, "f1")

    create_selected_walks_boxenplot(results, "accuracy")
    create_selected_walks_boxenplot(results, "f1")
    create_selected_walks_boxenplot(results, "f1", wanted=["3_4", "3_5", "4_3", "4_5", "5_3", "5_4"])
