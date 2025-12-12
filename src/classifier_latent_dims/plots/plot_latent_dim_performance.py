import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import List
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_results(base_path: Path, latent_dims: List[int], cancers: str, modalities: str, 
                walk_distance: int, walk_amount: int, iterations: int) -> pd.DataFrame:
    """Load and aggregate results across all iterations and latent dimensions"""
    
    all_results = []
    
    for latent_dim in latent_dims:
        for iteration in range(iterations):
            result_file = Path(base_path, str(latent_dim), cancers, modalities, 
                             f"{walk_distance}_{walk_amount}", str(iteration), "results.csv")
            
            if result_file.exists():
                df = pd.read_csv(result_file)
                df['latent_dim'] = latent_dim
                df['iteration'] = iteration
                all_results.append(df)
            else:
                print(f"Warning: Missing {result_file}")
    
    if not all_results:
        raise ValueError("No result files found!")
    
    return pd.concat(all_results, ignore_index=True)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results: compute mean, std, min, max across iterations"""
    
    # Separate per-cancer and overall results
    df_per_cancer = df[df['cancer'] != 'All'].copy()
    df_overall = df[df['cancer'] == 'All'].copy()
    
    # Aggregate per-cancer metrics (only accuracy, f1, precision, recall)
    if len(df_per_cancer) > 0:
        grouped_cancer = df_per_cancer.groupby(['latent_dim', 'cancer']).agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'f1': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        grouped_cancer.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in grouped_cancer.columns.values]
    else:
        grouped_cancer = pd.DataFrame()
    
    # Aggregate overall metrics (includes auc, mcc, balanced_accuracy)
    if len(df_overall) > 0:
        grouped_overall = df_overall.groupby(['latent_dim', 'cancer']).agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'f1': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std', 'min', 'max'],
            'auc': ['mean', 'std', 'min', 'max'],
            'mcc': ['mean', 'std', 'min', 'max'],
            'balanced_accuracy': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        grouped_overall.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                   for col in grouped_overall.columns.values]
    else:
        grouped_overall = pd.DataFrame()
    
    # Combine both
    if len(grouped_cancer) > 0 and len(grouped_overall) > 0:
        return pd.concat([grouped_cancer, grouped_overall], ignore_index=True)
    elif len(grouped_cancer) > 0:
        return grouped_cancer
    else:
        return grouped_overall


def plot_per_cancer_metric(df: pd.DataFrame, metric: str, save_path: Path, all_cancers: List[str]):
    """Plot a single metric with each cancer type as a separate line"""
    
    plt.figure(figsize=(14, 8))
    
    # Filter out 'All' for per-cancer plots
    df_cancers = df[df['cancer'] != 'All']
    
    # Plot each cancer type
    for cancer in all_cancers:
        cancer_data = df_cancers[df_cancers['cancer'] == cancer]
        if len(cancer_data) > 0:
            latent_dims = cancer_data['latent_dim'].values
            means = cancer_data[f'{metric}_mean'].values
            stds = cancer_data[f'{metric}_std'].values
            
            plt.plot(latent_dims, means, marker='o', linewidth=2, label=cancer, markersize=6)
            plt.fill_between(latent_dims, means - stds, means + stds, alpha=0.2)
    
    plt.xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'{metric.replace("_", " ").title()} by Cancer Type Across Latent Dimensions', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to steps of 50
    ax = plt.gca()
    if len(latent_dims) > 0:
        min_dim = min(latent_dims)
        max_dim = max(latent_dims)
        ax.set_xticks(range(int(min_dim), int(max_dim) + 1, 50))
    plt.tight_layout()
    
    plt.savefig(save_path / f'{metric}_per_cancer.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'{metric}_per_cancer.svg', dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / f'{metric}_per_cancer.png'}")


def plot_overall_metric(df: pd.DataFrame, metric: str, save_path: Path):
    """Plot overall performance (All cancers combined)"""
    
    plt.figure(figsize=(12, 7))
    
    # Filter for 'All' cancer type
    df_all = df[df['cancer'] == 'All']
    
    if len(df_all) > 0:
        latent_dims = df_all['latent_dim'].values
        means = df_all[f'{metric}_mean'].values
        stds = df_all[f'{metric}_std'].values
        mins = df_all[f'{metric}_min'].values
        maxs = df_all[f'{metric}_max'].values
        
        plt.plot(latent_dims, means, marker='o', linewidth=2.5, color='navy', 
                label='Mean', markersize=8)
        plt.fill_between(latent_dims, means - stds, means + stds, alpha=0.3, 
                        color='navy', label='±1 Std')
        plt.fill_between(latent_dims, mins, maxs, alpha=0.15, 
                        color='lightblue', label='Min-Max Range')
    
    plt.xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'Overall {metric.replace("_", " ").title()} Across Latent Dimensions', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to steps of 50
    ax = plt.gca()
    if len(latent_dims) > 0:
        min_dim = min(latent_dims)
        max_dim = max(latent_dims)
        ax.set_xticks(range(int(min_dim), int(max_dim) + 1, 50))
    
    plt.tight_layout()
    
    plt.savefig(save_path / f'{metric}_overall.png', dpi=500, bbox_inches='tight')
    plt.savefig(save_path / f'{metric}_overall.svg', dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / f'{metric}_overall.png'}")


def plot_all_metrics_panel(df: pd.DataFrame, save_path: Path, all_cancers: List[str]):
    """Create a panel plot: F1 per-cancer, Accuracy per-cancer, MCC overall, AUC overall"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Filter data
    df_cancers = df[df['cancer'] != 'All']
    df_overall = df[df['cancer'] == 'All']
    
    # Get latent dimension range for x-axis ticks
    all_latent_dims = sorted(df['latent_dim'].unique())
    if len(all_latent_dims) > 0:
        min_dim = int(min(all_latent_dims))
        max_dim = int(max(all_latent_dims))
        x_ticks = list(range(min_dim, max_dim + 1, 50))
    
    # Plot 1: F1 per cancer
    ax = axes[0]
    for cancer in all_cancers:
        cancer_data = df_cancers[df_cancers['cancer'] == cancer]
        if len(cancer_data) > 0:
            latent_dims = cancer_data['latent_dim'].values
            means = cancer_data['f1_mean'].values
            ax.plot(latent_dims, means, marker='o', linewidth=2, label=cancer, markersize=5)
    ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('F1 Score by Cancer Type', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(all_latent_dims) > 0:
        ax.set_xticks(x_ticks)
    
    # Plot 2: F1 overall
    ax = axes[1]
    if len(df_overall) > 0 and 'f1_mean' in df_overall.columns:
        latent_dims = df_overall['latent_dim'].values
        means = df_overall['f1_mean'].values
        stds = df_overall['f1_std'].values
        ax.plot(latent_dims, means, marker='o', linewidth=2.5, color='darkred', 
                label='Mean', markersize=7)
        ax.fill_between(latent_dims, means - stds, means + stds, alpha=0.3, color='darkred')
    ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('F1 Score (Overall)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(all_latent_dims) > 0:
        ax.set_xticks(x_ticks)
    
    # Plot 3: MCC overall
    ax = axes[2]
    if len(df_overall) > 0 and 'mcc_mean' in df_overall.columns:
        latent_dims = df_overall['latent_dim'].values
        means = df_overall['mcc_mean'].values
        stds = df_overall['mcc_std'].values
        ax.plot(latent_dims, means, marker='o', linewidth=2.5, color='navy', 
                label='Mean', markersize=7)
        ax.fill_between(latent_dims, means - stds, means + stds, alpha=0.3, color='navy')
    ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('MCC', fontsize=11, fontweight='bold')
    ax.set_title('Matthews Correlation Coefficient (Overall)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(all_latent_dims) > 0:
        ax.set_xticks(x_ticks)
    
    # Plot 4: AUC overall
    ax = axes[3]
    if len(df_overall) > 0 and 'auc_mean' in df_overall.columns:
        latent_dims = df_overall['latent_dim'].values
        means = df_overall['auc_mean'].values
        stds = df_overall['auc_std'].values
        ax.plot(latent_dims, means, marker='o', linewidth=2.5, color='darkgreen', 
                label='Mean', markersize=7)
        ax.fill_between(latent_dims, means - stds, means + stds, alpha=0.3, color='darkgreen')
    ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('Area Under ROC Curve (Overall)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(all_latent_dims) > 0:
        ax.set_xticks(x_ticks)
    ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('Area Under ROC Curve (Overall)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'all_metrics_panel.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'all_metrics_panel.png'}")


def save_aggregated_stats(df: pd.DataFrame, save_path: Path):
    """Save aggregated statistics to CSV"""
    
    output_file = save_path / 'aggregated_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved aggregated results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot latent dimension performance analysis')
    parser.add_argument('--base_path', '-b', type=str, 
                       default='results/classifier_latent_dims/classification',
                       help='Base path to classification results')
    parser.add_argument('--cancers', '-c', nargs='+', 
                       default=['BRCA', 'LUAD', 'STAD', 'BLCA', 'COAD', 'THCA'],
                       help='Cancer types')
    parser.add_argument('--modalities', '-m', type=str, default='mutations_rna',
                       help='Modality combination')
    parser.add_argument('--walk_distance', '-w', type=int, default=5,
                       help='Walk distance')
    parser.add_argument('--walk_amount', '-a', type=int, default=5,
                       help='Walk amount')
    parser.add_argument('--iterations', '-i', type=int, default=30,
                       help='Number of iterations')
    parser.add_argument('--latent_dims', '-ld', nargs='+', type=int,
                       default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
                       help='Latent dimensions to analyze')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.base_path)
    cancers_str = '_'.join(args.cancers)
    
    save_path = Path('figures', 'classifier_latent_dims', 'plots')
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {base_path}...")
    print(f"Cancers: {args.cancers}")
    print(f"Modalities: {args.modalities}")
    print(f"Latent dimensions: {args.latent_dims}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Load all results
    df_raw = load_results(base_path, args.latent_dims, cancers_str, args.modalities,
                         args.walk_distance, args.walk_amount, args.iterations)
    
    print(f"Loaded {len(df_raw)} result entries")
    print(f"Unique latent dimensions: {sorted(df_raw['latent_dim'].unique())}")
    print(f"Unique cancers: {sorted(df_raw['cancer'].unique())}")
    print()
    
    # Aggregate results
    print("Aggregating results across iterations...")
    df_agg = aggregate_results(df_raw)
    
    # Save aggregated statistics
    save_aggregated_stats(df_agg, save_path)
    
    # Create plots
    print("\nGenerating plots...")
    
    # Per-cancer metrics (only f1, accuracy, precision, recall)
    per_cancer_metrics = ['f1', 'accuracy', 'precision', 'recall']
    
    for metric in per_cancer_metrics:
        if f'{metric}_mean' in df_agg.columns:
            print(f"Plotting {metric} per cancer...")
            plot_per_cancer_metric(df_agg, metric, save_path, args.cancers)
    
    # Overall metrics (includes auc, mcc, balanced_accuracy)
    overall_metrics = ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc', 'balanced_accuracy']
    
    for metric in overall_metrics:
        # Check if metric exists in overall results
        df_overall = df_agg[df_agg['cancer'] == 'All']
        if len(df_overall) > 0 and f'{metric}_mean' in df_overall.columns:
            print(f"Plotting {metric} overall...")
            plot_overall_metric(df_agg, metric, save_path)
    
    # Panel plot with all metrics
    print("Creating panel plot...")
    plot_all_metrics_panel(df_agg, save_path, args.cancers)
    
    print("\n✅ All plots generated successfully!")
    print(f"📁 Output directory: {save_path}")


if __name__ == '__main__':
    main()
