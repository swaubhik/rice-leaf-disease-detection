"""Hyperparameter tuning script using Ray Tune for rice leaf disease detection."""
import argparse
import os
import json
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

from dataset import create_data_loaders
from models import create_model
from utils import AverageMeter


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return loss_meter.avg, 100. * correct / total


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Metrics
            loss_meter.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return loss_meter.avg, 100. * correct / total


def train_model_with_config(config: Dict, data_dir: str, num_classes: int, 
                            num_epochs: int = 20):
    """Training function for Ray Tune.
    
    Args:
        config: Hyperparameter configuration from Ray Tune
        data_dir: Path to dataset
        num_classes: Number of classes
        num_epochs: Number of epochs to train
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create data loaders with hyperparameters
    train_loader, val_loader, _ = create_data_loaders(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        num_workers=4,
        target_size=(config["image_size"], config["image_size"])
    )
    
    # Create model with hyperparameters
    model = create_model(
        num_classes=num_classes,
        model_name=config["model_name"],
        pretrained=True,
        dropout=config["dropout"],
        device=device
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    
    # Select optimizer based on config
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"]
        )
    
    # Setup learning rate scheduler
    if config["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif config["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("step_size", 10), gamma=0.1
        )
    else:  # reduce_on_plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    
    # Load checkpoint if resuming
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint_dict = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint_dict["model_state_dict"])
                optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if config["scheduler"] == "reduce_on_plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Save checkpoint
        checkpoint_dir_path = "."
        checkpoint_path = os.path.join(checkpoint_dir_path, "checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, checkpoint_path)
        
        # Report metrics to Ray Tune with checkpoint
        tune.report({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch
        })


def run_tuning(args):
    """Run hyperparameter tuning with Ray Tune."""
    
    # Convert paths to absolute paths (Ray Tune changes working directory)
    tune_dir_abs = os.path.abspath(args.tune_dir)
    data_dir_abs = os.path.abspath(args.data_dir)
    os.makedirs(tune_dir_abs, exist_ok=True)
    
    # Define search space
    config = {
        # Model architecture
        "model_name": tune.choice(["resnet50", "efficientnet_b0"]),
        "dropout": tune.uniform(0.2, 0.6),
        "image_size": tune.choice([224, 256, 288]),
        
        # Optimizer settings
        "optimizer": tune.choice(["adam", "adamw", "sgd"]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "momentum": tune.uniform(0.85, 0.95),  # For SGD
        
        # Training settings
        "batch_size": tune.choice([16, 24, 32, 48, 64]),
        "scheduler": tune.choice(["cosine", "step", "reduce_on_plateau"]),
        "step_size": tune.choice([5, 10, 15]),  # For StepLR
        "label_smoothing": tune.uniform(0.0, 0.2),
    }
    
    # Get number of classes from data
    from dataset import RiceLeafDataset
    import torchvision.transforms as transforms
    
    dataset = RiceLeafDataset(
        root_dir=data_dir_abs,
        split="train",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Data directory: {data_dir_abs}")
    
    # Setup scheduler based on algorithm choice
    # Ensure grace_period doesn't exceed max_epochs
    grace_period = min(args.grace_period, args.max_epochs)
    
    if args.search_alg == "asha":
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            max_t=args.max_epochs,
            grace_period=grace_period,
            reduction_factor=3
        )
        search_alg = None
    elif args.search_alg == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="epoch",
            metric="val_acc",
            mode="max",
            perturbation_interval=4,
            hyperparam_mutations={
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
            }
        )
        search_alg = None
    else:
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            max_t=args.max_epochs,
            grace_period=grace_period,
            reduction_factor=3
        )
        
        # Use Bayesian Optimization or HyperOpt
        if args.search_alg == "bayesopt":
            search_alg = BayesOptSearch(metric="val_acc", mode="max")
        else:  # hyperopt
            search_alg = HyperOptSearch(metric="val_acc", mode="max")
    
    # Setup reporter
    reporter = CLIReporter(
        metric_columns=["train_loss", "train_acc", "val_loss", "val_acc", "epoch"],
        max_progress_rows=20,
        max_report_frequency=30
    )
    
    # Run tuning
    print(f"\nStarting hyperparameter tuning with {args.search_alg}...")
    print(f"Number of trials: {args.num_samples}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"CPUs per trial: {args.cpus_per_trial}")
    print(f"GPUs per trial: {args.gpus_per_trial}")
    
    result = tune.run(
        partial(
            train_model_with_config,
            data_dir=data_dir_abs,
            num_classes=num_classes,
            num_epochs=args.max_epochs
        ),
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial
        },
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        storage_path=tune_dir_abs,
        name=args.experiment_name,
        resume=args.resume,
        raise_on_failed_trial=not args.ignore_failed,
        keep_checkpoints_num=3,
        checkpoint_score_attr="val_acc",
        verbose=1
    )
    
    # Get best trial (safe handling if some trials failed)
    try:
        best_trial = result.get_best_trial("val_acc", "max", "last")
        if best_trial is None:
            raise RuntimeError("No successful trials found")

        print("\n" + "="*80)
        print("Best trial config:")
        print(json.dumps(best_trial.config, indent=2))
        print(f"\nBest trial validation accuracy: {best_trial.last_result['val_acc']:.2f}%")
        print(f"Best trial validation loss: {best_trial.last_result['val_loss']:.4f}")
        print("="*80)
    except Exception as e:
        print("\nWarning: Could not determine a single best trial:", e)
        print("Showing top completed trials table instead.")
    
    # Save best config
    best_config_path = os.path.join(tune_dir_abs, args.experiment_name, "best_config.json")
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump({
            "config": best_trial.config,
            "val_acc": best_trial.last_result['val_acc'],
            "val_loss": best_trial.last_result['val_loss'],
            "train_acc": best_trial.last_result['train_acc'],
            "train_loss": best_trial.last_result['train_loss'],
        }, f, indent=2)
    print(f"\nBest config saved to {best_config_path}")
    
    # Print all trial results
    df = result.results_df
    df_sorted = df.sort_values("val_acc", ascending=False)
    print("\nTop 5 trials:")
    print(df_sorted[["val_acc", "val_loss", "train_acc", "config.model_name", 
                     "config.learning_rate", "config.batch_size"]].head())
    
    # Save results to CSV
    results_csv_path = os.path.join(tune_dir_abs, args.experiment_name, "all_results.csv")
    df.to_csv(results_csv_path, index=False)
    print(f"\nAll results saved to {results_csv_path}")
    
    return best_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for rice leaf disease detection")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to dataset directory")
    
    # Tuning arguments
    parser.add_argument("--search-alg", type=str, default="asha",
                        choices=["asha", "pbt", "hyperopt", "bayesopt"],
                        help="Search algorithm (asha=ASHA, pbt=PopulationBasedTraining, hyperopt=HyperOpt, bayesopt=BayesOpt)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of hyperparameter configurations to try")
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Maximum epochs per trial")
    parser.add_argument("--grace-period", type=int, default=5,
                        help="Minimum epochs before early stopping")
    
    # Resource arguments
    parser.add_argument("--cpus-per-trial", type=int, default=4,
                        help="CPUs allocated per trial")
    parser.add_argument("--gpus-per-trial", type=float, default=0.5,
                        help="GPUs allocated per trial (can be fractional)")
    
    # Output arguments
    parser.add_argument("--tune-dir", type=str, default="ray_results",
                        help="Directory to save Ray Tune results")
    parser.add_argument("--experiment-name", type=str, default="rice_disease_tune",
                        help="Name for this tuning experiment")
    parser.add_argument("--ignore-failed", action="store_true",
                        help="Do not raise an exception if some trials fail; continue and save results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous tuning run")
    
    args = parser.parse_args()
    
    # Run tuning
    best_trial = run_tuning(args)
    
    tune_dir_abs = os.path.abspath(args.tune_dir)

    # send complete  notification to discord webhook
    import requests

    webhook_url = "https://discord.com/api/webhooks/1446567027107041433/_VIfOVApFDsYSyL59cG6KimJ6vzpS8OnohL4f7fdMLjn_HJqQOZ67QR1xp_Tr3GKsbwh"
    
    try:
        message = {
            "content": f"🎉 Hyperparameter tuning complete!\n"
                      f"**Experiment:** {args.experiment_name}\n"
                      f"**Best Validation Accuracy:** {best_trial.last_result['val_acc']:.2f}%\n"
                      f"**Best Validation Loss:** {best_trial.last_result['val_loss']:.4f}\n"
                      f"**Model:** {best_trial.config['model_name']}\n"
                      f"**Learning Rate:** {best_trial.config['learning_rate']:.6f}\n"
                      f"**Batch Size:** {best_trial.config['batch_size']}\n"
                      f"Results saved to: `{tune_dir_abs}/{args.experiment_name}/`"
        }
        response = requests.post(webhook_url, json=message, timeout=10)
        if response.status_code == 204:
            print("\n✓ Discord notification sent successfully!")
        else:
            print(f"\n⚠ Discord notification failed: {response.status_code}")
    except Exception as e:
        print(f"\n⚠ Failed to send Discord notification: {e}")


    print("\n✓ Hyperparameter tuning complete!")
    print(f"  Use the best config from: {tune_dir_abs}/{args.experiment_name}/best_config.json")
    print(f"  To train with best config, update train.py arguments accordingly")
