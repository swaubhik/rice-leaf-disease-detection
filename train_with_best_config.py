#!/usr/bin/env python3
"""Train model using best hyperparameters from Ray Tune results."""
import argparse
import json
import os
import sys
import subprocess


def load_best_config(tune_results_dir: str, experiment_name: str) -> dict:
    """Load best configuration from Ray Tune results.
    
    Args:
        tune_results_dir: Directory where Ray Tune results are saved
        experiment_name: Name of the tuning experiment
        
    Returns:
        Dictionary with best configuration
    """
    config_path = os.path.join(tune_results_dir, experiment_name, "best_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Best config not found at {config_path}\n"
            f"Please run tune_hyperparameters.py first."
        )
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return data


def build_train_command(config: dict, args: argparse.Namespace) -> list:
    """Build training command from best config.
    
    Args:
        config: Best hyperparameter configuration
        args: Additional arguments from command line
        
    Returns:
        List of command arguments
    """
    cfg = config["config"]
    
    cmd = [
        "python", "train.py",
        "--data-dir", args.data_dir,
        "--model-name", cfg["model_name"],
        "--dropout", str(cfg["dropout"]),
        "--image-size", str(cfg["image_size"]),
        "--batch-size", str(cfg["batch_size"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--epochs", str(args.epochs),
        "--num-workers", str(args.num_workers),
        "--checkpoint-dir", args.checkpoint_dir,
        "--output-dir", args.output_dir,
    ]
    
    # Add optimizer-specific settings (handled via custom training if needed)
    # For now, train.py uses Adam by default
    
    # Add pretrained flag
    if args.pretrained:
        cmd.append("--pretrained")
    
    # Add freeze backbone epochs
    if args.freeze_backbone_epochs > 0:
        cmd.extend(["--freeze-backbone-epochs", str(args.freeze_backbone_epochs)])
    
    # Add resume if specified
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Train model with best hyperparameters from Ray Tune"
    )
    
    # Tuning results
    parser.add_argument("--tune-dir", type=str, default="ray_results",
                        help="Directory with Ray Tune results")
    parser.add_argument("--experiment-name", type=str, default="rice_disease_tune",
                        help="Name of tuning experiment")
    
    # Training arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (typically more than tuning)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save final models")
    
    # Additional options
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Number of epochs to freeze backbone")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    
    args = parser.parse_args()
    
    # Load best config
    print(f"Loading best config from {args.tune_dir}/{args.experiment_name}/")
    try:
        best_data = load_best_config(args.tune_dir, args.experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Display best configuration
    print("\n" + "="*80)
    print("Best Hyperparameters from Tuning:")
    print("="*80)
    for key, value in best_data["config"].items():
        print(f"  {key:20s}: {value}")
    print()
    print(f"  Validation Accuracy : {best_data['val_acc']:.2f}%")
    print(f"  Validation Loss     : {best_data['val_loss']:.4f}")
    print("="*80)
    
    # Build training command
    cmd = build_train_command(best_data, args)
    
    print("\nTraining Command:")
    print(" ".join(cmd))
    print()
    
    if args.dry_run:
        print("Dry run - not executing. Remove --dry-run to start training.")
        return
    
    # Confirm before training
    response = input("Start training with these hyperparameters? [y/N]: ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print("\nStarting training...")
    print("="*80 + "\n")
    
    # Execute training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
