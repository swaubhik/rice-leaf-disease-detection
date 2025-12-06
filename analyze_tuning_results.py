#!/usr/bin/env python3
"""Visualize and analyze Ray Tune hyperparameter tuning results."""
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(tune_dir: str, experiment_name: str) -> tuple:
    """Load tuning results and best config.
    
    Returns:
        Tuple of (best_config_dict, results_dataframe)
    """
    # Load best config
    config_path = os.path.join(tune_dir, experiment_name, "best_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Best config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        best_config = json.load(f)
    
    # Load all results CSV
    csv_path = os.path.join(tune_dir, experiment_name, "all_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    return best_config, df


def plot_hyperparameter_importance(df: pd.DataFrame, output_dir: str):
    """Plot correlation between hyperparameters and validation accuracy."""
    
    # Extract relevant columns
    hyperparam_cols = [col for col in df.columns if col.startswith('config.')]
    
    if len(hyperparam_cols) == 0:
        print("No hyperparameter columns found in results")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot top 6 hyperparameters
    for idx, col in enumerate(hyperparam_cols[:6]):
        ax = axes[idx]
        
        # Handle categorical vs numerical
        if df[col].dtype == 'object':
            # Categorical - use boxplot
            df_plot = df[[col, 'val_acc']].dropna()
            if len(df_plot) > 0:
                df_plot.boxplot(column='val_acc', by=col, ax=ax)
                ax.set_title(col.replace('config.', ''))
                ax.set_ylabel('Validation Accuracy (%)')
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')
        else:
            # Numerical - use scatter
            ax.scatter(df[col], df['val_acc'], alpha=0.6)
            ax.set_xlabel(col.replace('config.', ''))
            ax.set_ylabel('Validation Accuracy (%)')
            ax.set_title(col.replace('config.', ''))
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(hyperparam_cols), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Hyperparameter Impact on Validation Accuracy', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'hyperparameter_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved hyperparameter importance plot to {output_path}")
    plt.close()


def plot_trial_progression(df: pd.DataFrame, output_dir: str):
    """Plot how trials improved over time."""
    
    # Sort by trial start time if available
    if 'trial_id' in df.columns:
        df_sorted = df.sort_values('trial_id')
    else:
        df_sorted = df.copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation accuracy progression
    ax1.plot(df_sorted['val_acc'], marker='o', linewidth=1, markersize=4, alpha=0.7)
    ax1.axhline(y=df_sorted['val_acc'].max(), color='r', linestyle='--', 
                label=f'Best: {df_sorted["val_acc"].max():.2f}%')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy Across Trials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation loss progression
    ax2.plot(df_sorted['val_loss'], marker='o', linewidth=1, markersize=4, 
             alpha=0.7, color='orange')
    ax2.axhline(y=df_sorted['val_loss'].min(), color='r', linestyle='--',
                label=f'Best: {df_sorted["val_loss"].min():.4f}')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Across Trials')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'trial_progression.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved trial progression plot to {output_path}")
    plt.close()


def plot_top_configs(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    """Plot comparison of top N configurations."""
    
    # Get top N trials
    df_top = df.nlargest(top_n, 'val_acc')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df_top))
    ax.bar(x, df_top['val_acc'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Trial Rank')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title(f'Top {top_n} Configurations by Validation Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in x])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df_top['val_acc']):
        ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_configs.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved top configs plot to {output_path}")
    plt.close()


def print_summary(best_config: dict, df: pd.DataFrame):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("TUNING RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Trials: {len(df)}")
    print(f"Best Validation Accuracy: {best_config['val_acc']:.2f}%")
    print(f"Best Validation Loss: {best_config['val_loss']:.4f}")
    
    print(f"\nValidation Accuracy Statistics:")
    print(f"  Mean:   {df['val_acc'].mean():.2f}%")
    print(f"  Median: {df['val_acc'].median():.2f}%")
    print(f"  Std:    {df['val_acc'].std():.2f}%")
    print(f"  Min:    {df['val_acc'].min():.2f}%")
    print(f"  Max:    {df['val_acc'].max():.2f}%")
    
    print(f"\nBest Hyperparameters:")
    for key, value in best_config['config'].items():
        print(f"  {key:20s}: {value}")
    
    print("\n" + "="*80)
    
    # Top 5 configurations
    print("\nTop 5 Configurations:")
    print("-" * 80)
    
    df_sorted = df.sort_values('val_acc', ascending=False).head(5)
    
    for idx, row in df_sorted.iterrows():
        print(f"\nRank #{df_sorted.index.get_loc(idx) + 1}")
        print(f"  Val Acc: {row['val_acc']:.2f}% | Val Loss: {row['val_loss']:.4f}")
        
        # Print key hyperparameters
        for col in ['config.model_name', 'config.learning_rate', 'config.batch_size']:
            if col in row:
                key = col.replace('config.', '')
                print(f"  {key}: {row[col]}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Ray Tune results")
    
    parser.add_argument("--tune-dir", type=str, default="ray_results",
                        help="Directory with Ray Tune results")
    parser.add_argument("--experiment-name", type=str, default="rice_disease_tune",
                        help="Name of tuning experiment")
    parser.add_argument("--output-dir", type=str, default="tune_analysis",
                        help="Directory to save analysis plots")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top configs to plot")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.tune_dir}/{args.experiment_name}/")
    try:
        best_config, df = load_results(args.tune_dir, args.experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run tune_hyperparameters.py first.")
        return
    
    # Print summary
    print_summary(best_config, df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_hyperparameter_importance(df, args.output_dir)
    plot_trial_progression(df, args.output_dir)
    plot_top_configs(df, args.output_dir, args.top_n)
    
    print(f"\n✓ Analysis complete! Plots saved to {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  - {args.output_dir}/hyperparameter_importance.png")
    print(f"  - {args.output_dir}/trial_progression.png")
    print(f"  - {args.output_dir}/top_configs.png")


if __name__ == "__main__":
    main()
