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


def create_modality_comparison_barplot(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    """Create a grouped boxen plot comparing modalities (not cancers) between two models"""
    # Define modalities to include (exclude cancer types)
    modalities = ["Annotation", "Image", "RNA", "Mutation"]
    
    # Filter data to only include modalities
    modality_df = df[df["embedding"].isin(modalities)].copy()
    
    if modality_df.empty:
        logging.warning("No modality data found for comparison bar plot")
        return
    
    # Add readable model names
    modality_df["Model"] = modality_df["model"].map(model_names)
    
    # Calculate grid dimensions for separate plots per modality
    n_modalities = len(modalities)
    n_cols = 2
    n_rows = (n_modalities + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten() if n_modalities > 1 else [axes]
    
    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        modality_subset = modality_df[modality_df["embedding"] == modality]
        
        # Create the boxen plot with walk_distance on x-axis
        sns.boxenplot(
            data=modality_subset,
            x="walk_distance",
            y=metric.name,
            hue="Model",
            palette=["#5DA5DA", "#FAA43A"],  # Blue and orange for models
            ax=ax
        )
        
        ax.set_title(f"{modality}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Count", fontsize=10)
        ax.set_ylabel(metric.label if idx % n_cols == 0 else "", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(title="Model", fontsize=9)
        else:
            ax.legend().remove()
    
    # Hide unused subplots
    for idx in range(n_modalities, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"{metric.label} Comparison by Modality Across Sample Counts", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_modality_comparison_boxenplot.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {metric.name}_modality_comparison_boxenplot.png")


def create_modality_heatmap(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    """Create a heatmap showing performance of all modalities across sample counts for both models"""
    # Define modalities to include (exclude cancer types)
    modalities = ["Annotation", "Image", "RNA", "Mutation"]
    
    # Filter data to only include modalities
    modality_df = df[df["embedding"].isin(modalities)].copy()
    
    if modality_df.empty:
        logging.warning("No modality data found for heatmap")
        return
    
    # Add readable model names
    modality_df["Model"] = modality_df["model"].map(model_names)
    
    # Group by model, modality, and walk_distance, then calculate mean performance
    grouped = modality_df.groupby(["Model", "embedding", "walk_distance"])[metric.name].mean().reset_index()
    
    # Create separate heatmaps for each model side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for idx, model in enumerate(sorted(grouped["Model"].unique())):
        ax = axes[idx]
        model_data = grouped[grouped["Model"] == model]
        
        # Pivot data for heatmap: modalities as rows, sample counts as columns
        heatmap_data = model_data.pivot(index="embedding", columns="walk_distance", values=metric.name)
        
        # Reorder rows to match our preferred order
        heatmap_data = heatmap_data.reindex([m for m in modalities if m in heatmap_data.index])
        
        # Create heatmap with annotations
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={'label': metric.label},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_title(f"{model}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Count", fontsize=10)
        ax.set_ylabel("Modality" if idx == 0 else "", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    fig.suptitle(f"{metric.label} Heatmap: Modalities Across Sample Counts", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_modality_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {metric.name}_modality_heatmap.png")


def create_cancer_heatmap(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    """Create a heatmap showing performance across sample counts for BOTH modalities and cancer type embeddings.

    Rows: Modalities first (Annotation, Image, RNA, Mutation) then Cancer types (BRCA, LUAD, STAD, BLCA, COAD, THCA) if present.
    Columns: Sample counts (walk_distance)
    Cells: Mean metric value (e.g. F1)
    One heatmap per model side-by-side.
    """
    modalities = ["Annotation", "Image", "RNA", "Mutation"]
    cancer_types = ["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
    all_embeddings = modalities + cancer_types

    embedding_df = df[df["embedding"].isin(all_embeddings)].copy()
    if embedding_df.empty:
        logging.info("No modality or cancer embeddings found; skipping combined heatmap")
        return

    embedding_df["Model"] = embedding_df["model"].map(model_names)
    grouped = embedding_df.groupby(["Model", "embedding", "walk_distance"])[metric.name].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, model in enumerate(sorted(grouped["Model"].unique())):
        ax = axes[idx]
        model_data = grouped[grouped["Model"] == model]
        heatmap_data = model_data.pivot(index="embedding", columns="walk_distance", values=metric.name)
        # Reorder rows: modalities first then cancer types
        ordered_rows = [e for e in all_embeddings if e in heatmap_data.index]
        heatmap_data = heatmap_data.reindex(ordered_rows)

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={'label': metric.label},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        ax.set_title(f"{model}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Count", fontsize=10)
        ax.set_ylabel("Embedding" if idx == 0 else "", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle(f"{metric.label} Heatmap: Modalities + Cancer Types Across Sample Counts", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_cancer_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {metric.name}_cancer_heatmap.png (combined modalities + cancers)")


def create_modality_grouped_barchart(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    """Create a grouped bar chart showing all modalities across sample counts in one plot"""
    # Define modalities to include (exclude cancer types)
    modalities = ["Annotation", "Image", "RNA", "Mutation"]
    
    # Filter data to only include modalities
    modality_df = df[df["embedding"].isin(modalities)].copy()
    
    if modality_df.empty:
        logging.warning("No modality data found for grouped bar chart")
        return
    
    # Add readable model names
    modality_df["Model"] = modality_df["model"].map(model_names)
    
    # Group by model, modality, and walk_distance, then calculate mean and std
    grouped = modality_df.groupby(["Model", "embedding", "walk_distance"])[metric.name].agg(['mean', 'std']).reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get unique sample counts
    sample_counts = sorted(grouped["walk_distance"].unique())
    n_counts = len(sample_counts)
    n_modalities = len(modalities)
    
    # Set up bar positions
    x = range(n_counts)
    width = 0.08  # Width of each bar
    offset_per_group = width * n_modalities
    
    # Color palette for modalities
    modality_colors = {
        "Annotation": "#F17CB0",
        "Image": "#B276B2", 
        "RNA": "#60BD68",
        "Mutation": "#F15854"
    }
    
    # Plot bars for each model
    for model_idx, model in enumerate(sorted(grouped["Model"].unique())):
        model_data = grouped[grouped["Model"] == model]
        base_offset = model_idx * (offset_per_group + 0.02)  # Add small gap between model groups
        
        for mod_idx, modality in enumerate(modalities):
            mod_data = model_data[model_data["embedding"] == modality].sort_values("walk_distance")
            
            if mod_data.empty:
                continue
                
            positions = [xi + base_offset + mod_idx * width for xi in x]
            
            # Different patterns for different models (hatch for second model)
            hatch_pattern = None if model_idx == 0 else '//'
            
            ax.bar(
                positions,
                mod_data["mean"],
                width,
                yerr=mod_data["std"],
                label=f"{model} - {modality}" if model_idx < 1 else None,
                color=modality_colors[modality],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.85 if model_idx == 0 else 0.6,
                hatch=hatch_pattern,
                capsize=2
            )
    
    # Customize plot
    ax.set_xlabel("Sample Count", fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.label, fontsize=12, fontweight='bold')
    ax.set_title(f"{metric.label}: All Modalities Across Sample Counts", fontsize=14, fontweight='bold')
    ax.set_xticks([xi + (offset_per_group * len(grouped["Model"].unique()) + 0.02) / 2 - width for xi in x])
    ax.set_xticklabels(sample_counts)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create custom legend showing modalities
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=modality_colors[mod], edgecolor='black', label=mod) 
                      for mod in modalities]
    
    # Add model indicators to legend
    models_sorted = sorted(grouped["Model"].unique())
    legend_elements.append(Patch(facecolor='gray', alpha=0.85, label=f'{models_sorted[0]} (solid)'))
    if len(models_sorted) > 1:
        legend_elements.append(Patch(facecolor='gray', alpha=0.6, hatch='//', label=f'{models_sorted[1]} (hatched)'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_modality_grouped_barchart.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {metric.name}_modality_grouped_barchart.png")


def create_cancer_comparison_barplots(models: List[str], metric: Metric, df: pd.DataFrame, save_folder: Path):
    """Create individual bar plots with dots for each cancer type when comparing multi models"""
    # Check if we're comparing multi models
    is_multi_comparison = any("multi" in model for model in models)
    
    if not is_multi_comparison:
        logging.info("Not comparing multi models, skipping cancer-specific bar plots")
        return
    
    # Define cancer types
    cancer_types = ["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
    available_cancers = [cancer for cancer in cancer_types if cancer in df["embedding"].unique()]
    
    if not available_cancers:
        logging.warning("No cancer type data found for comparison bar plots")
        return
    
    # Add readable model names
    df_with_names = df.copy()
    df_with_names["Model"] = df_with_names["model"].map(model_names)
    
    # Calculate grid dimensions
    n_cancers = len(available_cancers)
    n_cols = 3
    n_rows = (n_cancers + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten() if n_cancers > 1 else [axes]
    
    for idx, cancer in enumerate(available_cancers):
        ax = axes[idx]
        
        # Filter data for this cancer type
        cancer_df = df_with_names[df_with_names["embedding"] == cancer].copy()
        
        # Group by model and walk_distance, calculate mean
        grouped_cancer_df = cancer_df.groupby(["Model", "walk_distance"])[metric.name].mean().reset_index()
        
        # Create bar plot
        sns.barplot(
            data=grouped_cancer_df,
            x="walk_distance",
            y=metric.name,
            hue="Model",
            palette=["#5DA5DA", "#FAA43A"],
            alpha=0.8,
            edgecolor="black",
            ax=ax
        )
        
        # Add strip plot to show individual data points
        sns.stripplot(
            data=cancer_df,
            x="walk_distance",
            y=metric.name,
            hue="Model",
            dodge=True,
            alpha=0.5,
            size=4,
            palette=["#5DA5DA", "#FAA43A"],
            ax=ax,
            legend=False
        )
        
        ax.set_title(f"{cancer}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Count", fontsize=10)
        ax.set_ylabel(metric.label if idx % n_cols == 0 else "", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Get legend handles and labels, remove duplicates
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            n_models = len(grouped_cancer_df["Model"].unique())
            ax.legend(handles[:n_models], labels[:n_models], title="Model", fontsize=9)
        else:
            ax.legend().remove()
    
    # Hide unused subplots
    for idx in range(n_cancers, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"{metric.label} Comparison by Cancer Type", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{metric.name}_cancer_comparison_barplots.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved {metric.name}_cancer_comparison_barplots.png")


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
    
    # Create modality comparison boxen plot
    create_modality_comparison_barplot(models, metric, df, save_folder)
    
    # Create modality heatmap
    create_modality_heatmap(models, metric, df, save_folder)
    
    # Create cancer heatmap (specific cancer embeddings across sample counts)
    create_cancer_heatmap(models, metric, df, save_folder)

    # Create modality grouped bar chart
    create_modality_grouped_barchart(models, metric, df, save_folder)
    
    # Create cancer-specific comparison bar plots (only for multi model comparisons)
    create_cancer_comparison_barplots(models, metric, df, save_folder)
