#!/usr/bin/env python3
"""
Comprehensive analysis comparing baseline model with tuned model.

This script analyzes:
- Model performance metrics (accuracy, loss)
- Hyperparameter differences
- Training efficiency
- Generalization capability
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from typing import Dict, Any
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_baseline_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load baseline model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract relevant metrics
    baseline_info = {
        'epoch': ckpt.get('epoch', 'N/A'),
        'val_accuracy': ckpt.get('val_accuracy', ckpt.get('val_acc', 0.0)),
        'val_loss': ckpt.get('val_loss', 0.0),
        'train_accuracy': ckpt.get('train_accuracy', ckpt.get('train_acc', 0.0)),
        'train_loss': ckpt.get('train_loss', 0.0),
        'model_name': ckpt.get('model_name', 'Unknown'),
        'config': ckpt.get('config', {})
    }
    
    return baseline_info


def load_tuned_model_info(best_config_path: str, results_csv_path: str) -> Dict[str, Any]:
    """Load tuned model information."""
    # Load best config
    with open(best_config_path, 'r') as f:
        best_config_data = json.load(f)
    
    # Load all results
    results_df = pd.read_csv(results_csv_path)
    
    # Get best trial (sorted by validation accuracy)
    best_trial = results_df.sort_values('val_acc', ascending=False).iloc[0]
    
    tuned_info = {
        'config': best_config_data.get('config', {}),
        'val_accuracy': best_config_data.get('val_acc', 0.0),
        'val_loss': best_config_data.get('val_loss', 0.0),
        'train_accuracy': best_config_data.get('train_acc', 0.0),
        'train_loss': best_config_data.get('train_loss', 0.0),
        'epoch': best_trial['epoch'] if 'epoch' in best_trial else 'N/A',
        'num_trials': len(results_df),
        'results_df': results_df
    }
    
    return tuned_info


def create_comparison_table(baseline: Dict, tuned: Dict) -> pd.DataFrame:
    """Create a comparison table of key metrics."""
    
    metrics = {
        'Metric': [
            'Validation Accuracy (%)',
            'Validation Loss',
            'Training Accuracy (%)',
            'Training Loss',
            'Epochs Trained',
            'Model Architecture',
            'Batch Size',
            'Learning Rate',
            'Dropout',
            'Image Size',
            'Optimizer',
            'Scheduler',
            'Label Smoothing'
        ],
        'Baseline Model': [
            f"{baseline['val_accuracy']:.2f}",
            f"{baseline['val_loss']:.4f}",
            f"{baseline['train_accuracy']:.2f}",
            f"{baseline['train_loss']:.4f}",
            baseline['epoch'],
            baseline.get('config', {}).get('model_name', baseline.get('model_name', 'N/A')),
            baseline.get('config', {}).get('batch_size', 'N/A'),
            baseline.get('config', {}).get('learning_rate', 'N/A'),
            baseline.get('config', {}).get('dropout', 'N/A'),
            baseline.get('config', {}).get('image_size', 'N/A'),
            baseline.get('config', {}).get('optimizer', 'N/A'),
            baseline.get('config', {}).get('scheduler', 'N/A'),
            baseline.get('config', {}).get('label_smoothing', 'N/A'),
        ],
        'Tuned Model': [
            f"{tuned['val_accuracy']:.2f}",
            f"{tuned['val_loss']:.4f}",
            f"{tuned['train_accuracy']:.2f}",
            f"{tuned['train_loss']:.4f}",
            tuned['epoch'],
            tuned['config'].get('model_name', 'N/A'),
            tuned['config'].get('batch_size', 'N/A'),
            tuned['config'].get('learning_rate', 'N/A'),
            tuned['config'].get('dropout', 'N/A'),
            tuned['config'].get('image_size', 'N/A'),
            tuned['config'].get('optimizer', 'N/A'),
            tuned['config'].get('scheduler', 'N/A'),
            tuned['config'].get('label_smoothing', 'N/A'),
        ]
    }
    
    # Calculate improvements
    improvements = []
    for i, metric_name in enumerate(metrics['Metric']):
        baseline_val = metrics['Baseline Model'][i]
        tuned_val = metrics['Tuned Model'][i]
        
        if metric_name in ['Validation Accuracy (%)', 'Training Accuracy (%)']:
            try:
                diff = float(tuned_val) - float(baseline_val)
                improvements.append(f"{diff:+.2f}%" if diff != 0 else "=")
            except:
                improvements.append("N/A")
        elif metric_name in ['Validation Loss', 'Training Loss']:
            try:
                diff = float(tuned_val) - float(baseline_val)
                improvements.append(f"{diff:+.4f}" if diff != 0 else "=")
            except:
                improvements.append("N/A")
        else:
            improvements.append("" if baseline_val == tuned_val else "Changed")
    
    metrics['Improvement'] = improvements
    
    return pd.DataFrame(metrics)


def plot_performance_comparison(baseline: Dict, tuned: Dict, output_dir: Path):
    """Create performance comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline vs Tuned Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    categories = ['Training Accuracy', 'Validation Accuracy']
    baseline_accs = [baseline['train_accuracy'], baseline['val_accuracy']]
    tuned_accs = [tuned['train_accuracy'], tuned['val_accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, tuned_accs, width, label='Tuned', alpha=0.8, color='#2ecc71')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim([min(baseline_accs + tuned_accs) - 1, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Loss Comparison
    ax2 = axes[0, 1]
    categories = ['Training Loss', 'Validation Loss']
    baseline_losses = [baseline['train_loss'], baseline['val_loss']]
    tuned_losses = [tuned['train_loss'], tuned['val_loss']]
    
    bars1 = ax2.bar(x - width/2, baseline_losses, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax2.bar(x + width/2, tuned_losses, width, label='Tuned', alpha=0.8, color='#2ecc71')
    
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Overfitting Analysis (Train vs Val gap)
    ax3 = axes[1, 0]
    models = ['Baseline', 'Tuned']
    train_val_gap_acc = [
        baseline['train_accuracy'] - baseline['val_accuracy'],
        tuned['train_accuracy'] - tuned['val_accuracy']
    ]
    train_val_gap_loss = [
        baseline['val_loss'] - baseline['train_loss'],
        tuned['val_loss'] - tuned['train_loss']
    ]
    
    x_pos = np.arange(len(models))
    bars = ax3.bar(x_pos, train_val_gap_acc, alpha=0.8, color=['#e74c3c', '#f39c12'])
    ax3.set_ylabel('Train-Val Accuracy Gap (%)', fontsize=12)
    ax3.set_title('Generalization Gap (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
    
    # 4. Key Hyperparameter Differences
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = "Key Hyperparameter Changes:\n\n"
    
    config_keys = ['batch_size', 'learning_rate', 'dropout', 'image_size', 'label_smoothing']
    for key in config_keys:
        baseline_val = baseline.get('config', {}).get(key, 'N/A')
        tuned_val = tuned['config'].get(key, 'N/A')
        
        if baseline_val != tuned_val:
            summary_text += f"• {key.replace('_', ' ').title()}:\n"
            summary_text += f"  {baseline_val} → {tuned_val}\n\n"
    
    # Add performance summary
    acc_improvement = tuned['val_accuracy'] - baseline['val_accuracy']
    loss_improvement = baseline['val_loss'] - tuned['val_loss']
    
    summary_text += f"\nPerformance Summary:\n"
    summary_text += f"• Val Accuracy: {acc_improvement:+.2f}%\n"
    summary_text += f"• Val Loss: {loss_improvement:+.4f}\n"
    summary_text += f"• Trials Explored: {tuned['num_trials']}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_trial_distribution(tuned: Dict, output_dir: Path):
    """Plot distribution of validation accuracies across all trials."""
    
    results_df = tuned['results_df']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Hyperparameter Tuning Trial Analysis', fontsize=16, fontweight='bold')
    
    # 1. Distribution of validation accuracies
    ax1 = axes[0]
    ax1.hist(results_df['val_acc'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(tuned['val_accuracy'], color='red', linestyle='--', linewidth=2, label='Best Trial')
    ax1.set_xlabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_ylabel('Number of Trials', fontsize=12)
    ax1.set_title(f'Distribution of Val Accuracy Across {tuned["num_trials"]} Trials', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top 10 trials
    ax2 = axes[1]
    top_trials = results_df.nlargest(10, 'val_acc')
    trial_names = [f"Trial {i+1}" for i in range(len(top_trials))]
    
    bars = ax2.barh(trial_names, top_trials['val_acc'].values, alpha=0.8, color='#2ecc71')
    bars[0].set_color('#e74c3c')  # Highlight best trial
    
    ax2.set_xlabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Top 10 Trial Results', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trial_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved trial distribution to {output_dir / 'trial_distribution.png'}")
    plt.close()


def generate_markdown_report(baseline: Dict, tuned: Dict, comparison_df: pd.DataFrame, output_dir: Path):
    """Generate a markdown report summarizing the analysis."""
    
    report = f"""# Model Comparison Report
## Rice Leaf Disease Detection: Baseline vs Tuned Model

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report compares the performance of the baseline model with the hyperparameter-tuned model for rice leaf disease detection.

### Key Findings

- **Validation Accuracy Improvement:** {tuned['val_accuracy'] - baseline['val_accuracy']:+.2f}%
- **Validation Loss Improvement:** {baseline['val_loss'] - tuned['val_loss']:+.4f}
- **Total Trials Explored:** {tuned['num_trials']}
- **Best Model Configuration:** {tuned['config'].get('model_name', 'N/A')}

---

## Detailed Metrics Comparison

{comparison_df.to_markdown(index=False)}

---

## Performance Analysis

### Accuracy Comparison
- **Baseline Validation Accuracy:** {baseline['val_accuracy']:.2f}%
- **Tuned Validation Accuracy:** {tuned['val_accuracy']:.2f}%
- **Improvement:** {tuned['val_accuracy'] - baseline['val_accuracy']:+.2f} percentage points

### Loss Comparison
- **Baseline Validation Loss:** {baseline['val_loss']:.4f}
- **Tuned Validation Loss:** {tuned['val_loss']:.4f}
- **Improvement:** {baseline['val_loss'] - tuned['val_loss']:+.4f}

### Generalization Analysis
- **Baseline Train-Val Gap:** {baseline['train_accuracy'] - baseline['val_accuracy']:.2f}%
- **Tuned Train-Val Gap:** {tuned['train_accuracy'] - tuned['val_accuracy']:.2f}%
- **Better Generalization:** {"Tuned" if (tuned['train_accuracy'] - tuned['val_accuracy']) < (baseline['train_accuracy'] - baseline['val_accuracy']) else "Baseline"}

---

## Hyperparameter Changes

### Model Configuration

| Hyperparameter | Baseline | Tuned | Change |
|----------------|----------|-------|--------|
| Model | {baseline.get('config', {}).get('model_name', 'N/A')} | {tuned['config'].get('model_name', 'N/A')} | {"✓" if baseline.get('config', {}).get('model_name') != tuned['config'].get('model_name') else "="} |
| Batch Size | {baseline.get('config', {}).get('batch_size', 'N/A')} | {tuned['config'].get('batch_size', 'N/A')} | {"✓" if baseline.get('config', {}).get('batch_size') != tuned['config'].get('batch_size') else "="} |
| Learning Rate | {baseline.get('config', {}).get('learning_rate', 'N/A')} | {tuned['config'].get('learning_rate', 'N/A')} | {"✓" if baseline.get('config', {}).get('learning_rate') != tuned['config'].get('learning_rate') else "="} |
| Dropout | {baseline.get('config', {}).get('dropout', 'N/A')} | {tuned['config'].get('dropout', 'N/A')} | {"✓" if baseline.get('config', {}).get('dropout') != tuned['config'].get('dropout') else "="} |
| Image Size | {baseline.get('config', {}).get('image_size', 'N/A')} | {tuned['config'].get('image_size', 'N/A')} | {"✓" if baseline.get('config', {}).get('image_size') != tuned['config'].get('image_size') else "="} |
| Label Smoothing | {baseline.get('config', {}).get('label_smoothing', 'N/A')} | {tuned['config'].get('label_smoothing', 'N/A')} | {"✓" if baseline.get('config', {}).get('label_smoothing') != tuned['config'].get('label_smoothing') else "="} |

---

## Conclusions

### Strengths of Tuned Model
1. {"Higher validation accuracy" if tuned['val_accuracy'] > baseline['val_accuracy'] else "Maintained high accuracy"}
2. {"Better generalization (lower train-val gap)" if (tuned['train_accuracy'] - tuned['val_accuracy']) < (baseline['train_accuracy'] - baseline['val_accuracy']) else "Strong performance"}
3. {"Optimized hyperparameters through systematic search"}

### Recommendations
1. **Deploy:** {"Tuned model" if tuned['val_accuracy'] >= baseline['val_accuracy'] else "Baseline model"} shows better overall performance
2. **Monitor:** Track real-world performance to validate improvements
3. **Future Work:** Consider ensemble methods or additional data augmentation

---

## Visualizations

See accompanying images:
- `model_comparison.png` - Side-by-side performance comparison
- `trial_distribution.png` - Distribution of trial results
- `hyperparameter_importance.png` - Impact of different hyperparameters
- `trial_progression.png` - Learning curves across trials

---

*Report generated by compare_models.py*
"""
    
    report_path = output_dir / 'MODEL_COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nMarkdown report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and tuned models')
    parser.add_argument('--baseline-checkpoint', type=str, 
                       default='checkpoints/best_model_checkpoint.pth',
                       help='Path to baseline model checkpoint')
    parser.add_argument('--tuned-config', type=str,
                       default='ray_results/rice_disease_tune/best_config.json',
                       help='Path to tuned model best config')
    parser.add_argument('--results-csv', type=str,
                       default='ray_results/rice_disease_tune/all_results.csv',
                       help='Path to all results CSV')
    parser.add_argument('--output-dir', type=str,
                       default='model_comparison_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Rice Leaf Disease Detection: Model Comparison Analysis")
    print("=" * 80)
    
    # Load baseline model
    print("\n📊 Loading baseline model...")
    baseline = load_baseline_checkpoint(args.baseline_checkpoint)
    print(f"   Baseline Val Accuracy: {baseline['val_accuracy']:.2f}%")
    
    # Load tuned model
    print("\n🔧 Loading tuned model...")
    tuned = load_tuned_model_info(args.tuned_config, args.results_csv)
    print(f"   Tuned Val Accuracy: {tuned['val_accuracy']:.2f}%")
    print(f"   Total Trials: {tuned['num_trials']}")
    
    # Create comparison table
    print("\n📋 Creating comparison table...")
    comparison_df = create_comparison_table(baseline, tuned)
    print("\n" + "=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Save comparison table
    comparison_df.to_csv(output_dir / 'comparison_table.csv', index=False)
    print(f"\n✓ Comparison table saved to {output_dir / 'comparison_table.csv'}")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    plot_performance_comparison(baseline, tuned, output_dir)
    plot_trial_distribution(tuned, output_dir)
    
    # Generate markdown report
    print("\n📝 Generating markdown report...")
    generate_markdown_report(baseline, tuned, comparison_df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    acc_diff = tuned['val_accuracy'] - baseline['val_accuracy']
    loss_diff = baseline['val_loss'] - tuned['val_loss']
    
    print(f"\n✓ Validation Accuracy Change: {acc_diff:+.2f}%")
    print(f"✓ Validation Loss Change: {loss_diff:+.4f}")
    print(f"✓ Total Hyperparameter Trials: {tuned['num_trials']}")
    
    if acc_diff > 0:
        print(f"\n🎉 The tuned model shows IMPROVED performance!")
    elif abs(acc_diff) < 0.1:
        print(f"\n✓ The tuned model maintains similar performance with optimized hyperparameters")
    else:
        print(f"\n⚠️  The baseline model performed slightly better - consider further tuning")
    
    print(f"\n📁 All results saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
