import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def collect_results(base_folder: Path, cancers: str, modalities: str, walk_distance: int, 
                    walk_amount: int, latent_dims: list, iterations: range) -> pd.DataFrame:
    """
    Collect all results.csv files for each latent dimension and iteration.
    
    Args:
        base_folder: Base results folder
        cancers: Cancer types string (e.g., "BRCA_LUAD_STAD_BLCA_COAD_THCA")
        modalities: Modalities string (e.g., "mutations_rna")
        walk_distance: Walk distance parameter
        walk_amount: Walk amount parameter
        latent_dims: List of latent dimensions to process
        iterations: Range of iterations to process
        
    Returns:
        DataFrame with all results aggregated
    """
    all_results = []
    
    for latent_dim in latent_dims:
        logging.info(f"Processing latent_dim: {latent_dim}")
        
        for iteration in iterations:
            result_file = Path(
                base_folder, cancers, modalities, str(latent_dim), 
                f"{walk_distance}_{walk_amount}", str(iteration), "results.csv"
            )
            
            if result_file.exists():
                try:
                    df = pd.read_csv(result_file)
                    # Add metadata
                    df['latent_dim'] = latent_dim
                    df['iteration'] = iteration
                    df['walk_distance'] = walk_distance
                    df['walk_amount'] = walk_amount
                    all_results.append(df)
                except Exception as e:
                    logging.warning(f"Failed to read {result_file}: {e}")
            else:
                logging.warning(f"File not found: {result_file}")
    
    if not all_results:
        raise ValueError("No results files found!")
    
    return pd.concat(all_results, ignore_index=True)


def aggregate_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results by latent dimension.
    Calculate mean, std, min, max for each metric.
    
    Returns:
        Tuple of (overall_stats, per_cancer_stats)
    """
    # Overall stats (using "All" cancer type)
    df_all = df[df['cancer'] == 'All'].copy()
    
    overall_stats = df_all.groupby('latent_dim').agg({
        'f1': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'auc': ['mean', 'std', 'min', 'max'],
        'mcc': ['mean', 'std', 'min', 'max'],
        'balanced_accuracy': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    overall_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in overall_stats.columns.values]
    
    # Per-cancer stats (excluding "All")
    df_per_cancer = df[df['cancer'] != 'All'].copy()
    
    per_cancer_stats = df_per_cancer.groupby(['latent_dim', 'cancer']).agg({
        'f1': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    per_cancer_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in per_cancer_stats.columns.values]
    
    return overall_stats, per_cancer_stats


def plot_per_cancer_performance(per_cancer_df: pd.DataFrame, metric: str, save_folder: Path,
                                cancers: str, modalities: str, walk_distance: int, walk_amount: int):
    """
    Plot performance metric vs latent dimension for each cancer type separately.
    
    Args:
        per_cancer_df: Per-cancer aggregated results DataFrame
        metric: Metric to plot (e.g., 'f1', 'accuracy')
        save_folder: Folder to save the plot
        cancers: Cancer types string
        modalities: Modalities string
        walk_distance: Walk distance parameter
        walk_amount: Walk amount parameter
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique cancers
    cancer_types = sorted(per_cancer_df['cancer'].unique())
    
    # Color palette
    colors = sns.color_palette("husl", len(cancer_types))
    
    # Plot each cancer type
    for idx, cancer in enumerate(cancer_types):
        cancer_data = per_cancer_df[per_cancer_df['cancer'] == cancer]
        
        x = cancer_data['latent_dim']
        y_mean = cancer_data[f'{metric}_mean']
        y_std = cancer_data[f'{metric}_std']
        
        # Line plot with markers
        ax.plot(x, y_mean, marker='o', markersize=7, linewidth=2.5, 
                label=cancer, color=colors[idx], alpha=0.8)
        
        # Error bars (standard deviation) - lighter
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                         alpha=0.15, color=colors[idx])
    
    # Styling
    ax.set_xlabel('Latent Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric.upper()} Performance vs Latent Dimension (Per Cancer Type)\n'
                f'Modalities: {modalities.replace("_", ", ")} | Walk: {walk_distance}_{walk_amount}',
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend - outside plot area
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5), 
              framealpha=0.9, title='Cancer Type', title_fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_folder, f'{metric}_per_cancer_vs_latent_dim_{cancers}_{modalities}_{walk_distance}_{walk_amount}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved per-cancer plot to {save_path}")
    plt.close()


def plot_latent_dim_performance(agg_df: pd.DataFrame, metric: str, save_folder: Path,
                                cancers: str, modalities: str, walk_distance: int, walk_amount: int):
    """
    Plot performance metric vs latent dimension with error bars (overall performance).
    
    Args:
        agg_df: Aggregated results DataFrame
        metric: Metric to plot (e.g., 'f1', 'accuracy')
        save_folder: Folder to save the plot
        cancers: Cancer types string
        modalities: Modalities string
        walk_distance: Walk distance parameter
        walk_amount: Walk amount parameter
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract columns
    x = agg_df['latent_dim']
    y_mean = agg_df[f'{metric}_mean']
    y_std = agg_df[f'{metric}_std']
    y_min = agg_df[f'{metric}_min']
    y_max = agg_df[f'{metric}_max']
    
    # Main line plot with markers
    ax.plot(x, y_mean, marker='o', markersize=8, linewidth=2, 
            label=f'{metric.upper()} (mean)', color='#2E86AB')
    
    # Error bars (standard deviation)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                     alpha=0.2, color='#2E86AB', label='±1 std')
    
    # Min/max range as lighter shading
    ax.fill_between(x, y_min, y_max, 
                     alpha=0.1, color='#2E86AB', label='Min-Max range')
    
    # Styling
    ax.set_xlabel('Latent Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric.upper()} Performance vs Latent Dimension (Overall)\n'
                f'Cancers: {cancers.replace("_", ", ")} | Modalities: {modalities.replace("_", ", ")} | '
                f'Walk: {walk_distance}_{walk_amount}',
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Add value annotations for mean
    for i, (xi, yi) in enumerate(zip(x, y_mean)):
        ax.annotate(f'{yi:.3f}', 
                   xy=(xi, yi), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_folder, f'{metric}_overall_vs_latent_dim_{cancers}_{modalities}_{walk_distance}_{walk_amount}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved overall plot to {save_path}")
    plt.close()


def plot_all_metrics(agg_df: pd.DataFrame, save_folder: Path, 
                    cancers: str, modalities: str, walk_distance: int, walk_amount: int):
    """
    Create a multi-panel plot with all metrics.
    """
    metrics = ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc', 'balanced_accuracy']
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = agg_df['latent_dim']
        y_mean = agg_df[f'{metric}_mean']
        y_std = agg_df[f'{metric}_std']
        
        # Line plot with error bars
        ax.plot(x, y_mean, marker='o', markersize=6, linewidth=2, color='#2E86AB')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Performance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Remove extra subplots
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'All Metrics vs Latent Dimension\n'
                f'Cancers: {cancers.replace("_", ", ")} | Modalities: {modalities.replace("_", ", ")} | '
                f'Walk: {walk_distance}_{walk_amount}',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    save_path = Path(save_folder, f'all_metrics_vs_latent_dim_{cancers}_{modalities}_{walk_distance}_{walk_amount}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved multi-metric plot to {save_path}")
    plt.close()


def save_aggregated_results(agg_df: pd.DataFrame, save_folder: Path,
                           cancers: str, modalities: str, walk_distance: int, walk_amount: int):
    """
    Save aggregated statistics to CSV.
    """
    save_path = Path(save_folder, f'aggregated_results_{cancers}_{modalities}_{walk_distance}_{walk_amount}.csv')
    agg_df.to_csv(save_path, index=False)
    logging.info(f"Saved aggregated results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latent dimension performance across iterations")
    parser.add_argument("--base_folder", "-b", type=str, 
                       default="results/classifier_latent_dims/classification",
                       help="Base folder containing classification results")
    parser.add_argument("--save_folder", "-s", type=str,
                       default="results/classifier_latent_dims/plots",
                       help="Folder to save plots")
    parser.add_argument("--cancers", "-c", type=str,
                       default="BRCA_LUAD_STAD_BLCA_COAD_THCA",
                       help="Cancer types (underscore-separated)")
    parser.add_argument("--modalities", "-m", type=str,
                       default="mutations_rna",
                       help="Modalities (underscore-separated)")
    parser.add_argument("--walk_distance", "-w", type=int, default=5,
                       help="Walk distance parameter")
    parser.add_argument("--walk_amount", "-a", type=int, default=5,
                       help="Walk amount parameter")
    parser.add_argument("--latent_dims", "-ld", nargs="+", type=int,
                       default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
                       help="List of latent dimensions to process")
    parser.add_argument("--num_iterations", "-n", type=int, default=30,
                       help="Number of iterations to aggregate")
    
    args = parser.parse_args()
    
    # Create save folder
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect all results
    logging.info("Collecting results from all iterations and latent dimensions...")
    all_results = collect_results(
        base_folder=Path(args.base_folder),
        cancers=args.cancers,
        modalities=args.modalities,
        walk_distance=args.walk_distance,
        walk_amount=args.walk_amount,
        latent_dims=args.latent_dims,
        iterations=range(args.num_iterations)
    )
    
    logging.info(f"Collected {len(all_results)} result entries")
    
    # Aggregate results
    logging.info("Aggregating results by latent dimension...")
    overall_stats, per_cancer_stats = aggregate_results(all_results)
    
    logging.info(f"Aggregated results for {len(overall_stats)} latent dimensions")
    print("\nOverall Aggregated Results:")
    print(overall_stats[['latent_dim', 'f1_mean', 'f1_std', 'accuracy_mean', 'accuracy_std']])
    
    print("\nPer-Cancer Results (sample):")
    print(per_cancer_stats[['latent_dim', 'cancer', 'f1_mean', 'f1_std']].head(20))
    
    # Save aggregated results
    save_aggregated_results(overall_stats, save_folder, args.cancers, args.modalities, 
                          args.walk_distance, args.walk_amount)
    
    # Save per-cancer results
    per_cancer_save_path = Path(save_folder, f'per_cancer_results_{args.cancers}_{args.modalities}_{args.walk_distance}_{args.walk_amount}.csv')
    per_cancer_stats.to_csv(per_cancer_save_path, index=False)
    logging.info(f"Saved per-cancer results to {per_cancer_save_path}")
    
    # Plot individual metrics (overall)
    logging.info("Creating overall metric plots...")
    for metric in ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc', 'balanced_accuracy']:
        plot_latent_dim_performance(overall_stats, metric, save_folder, 
                                   args.cancers, args.modalities, 
                                   args.walk_distance, args.walk_amount)
    
    # Plot per-cancer metrics
    logging.info("Creating per-cancer metric plots...")
    for metric in ['f1', 'accuracy', 'precision', 'recall']:
        plot_per_cancer_performance(per_cancer_stats, metric, save_folder,
                                   args.cancers, args.modalities,
                                   args.walk_distance, args.walk_amount)
    
    # Plot all metrics together
    logging.info("Creating multi-metric plot...")
    plot_all_metrics(overall_stats, save_folder, args.cancers, args.modalities,
                    args.walk_distance, args.walk_amount)
    
    logging.info("All plots created successfully!")
