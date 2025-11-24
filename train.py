"""Training script for rice leaf disease detection."""
import argparse
import os
import json
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import create_data_loaders
from models import create_model
from utils import AverageMeter, save_checkpoint, load_checkpoint


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in pbar:
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
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc:.2f}%"})
    
    return loss_meter.avg, 100. * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    
    with torch.no_grad():
        for images, labels in pbar:
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
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc:.2f}%"})
    
    return loss_meter.avg, 100. * correct / total


def plot_training_history(history: Dict, save_path: str):
    """Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def main(args):
    """Main training function."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, classes = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.image_size, args.image_size)
    )
    
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Save class names
    class_names_path = os.path.join(args.output_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"Class names saved to {class_names_path}")
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        dropout=args.dropout,
        device=device
    )
    
    # Optionally freeze backbone for first few epochs
    if args.freeze_backbone_epochs > 0:
        print(f"Freezing backbone for first {args.freeze_backbone_epochs} epochs")
        model.freeze_backbone()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, optimizer)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            history = checkpoint.get('history', history)
            print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone after specified epochs
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            print(f"\nUnfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()
            # Recreate optimizer with all parameters
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_acc': val_acc,
                'history': history,
                'model_name': args.model_name,
                'num_classes': num_classes,
                'classes': classes
            },
            is_best=is_best,
            checkpoint_dir=args.checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch}.pth"
        )
        
        if is_best:
            print(f"  New best model! Accuracy: {best_acc:.2f}%")
            # Also save to output directory
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rice leaf disease detection model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to dataset directory")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3"],
                        help="Backbone model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Number of epochs to freeze backbone (0 to disable)")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save final models")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    main(args)
