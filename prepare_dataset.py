"""Script to split dataset into train and validation sets."""
import os
import shutil
import random
from pathlib import Path
import argparse


def split_dataset(
    source_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42
):
    """Split dataset into train and validation sets.
    
    Args:
        source_dir: Source directory containing class folders
        output_dir: Output directory for train/val split
        val_split: Validation split ratio (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {source_dir}")
    
    print(f"Found {len(class_dirs)} classes:")
    for class_dir in class_dirs:
        print(f"  - {class_dir.name}")
    
    print(f"\nSplitting dataset (train: {1-val_split:.1%}, val: {val_split:.1%})...")
    
    total_train = 0
    total_val = 0
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_name}")
            continue
        
        # Shuffle with seed for reproducibility
        random.seed(seed)
        random.shuffle(image_files)
        
        # Split files
        val_size = int(len(image_files) * val_split)
        val_files = image_files[:val_size]
        train_files = image_files[val_size:]
        
        # Create class directories
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy training files
        for file in train_files:
            shutil.copy2(file, train_class_dir / file.name)
        
        # Copy validation files
        for file in val_files:
            shutil.copy2(file, val_class_dir / file.name)
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        print(f"  {class_name}: {len(train_files)} train, {len(val_files)} val")
    
    print(f"\nDataset split complete!")
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val sets")
    parser.add_argument("--source-dir", type=str, default="Augmented Images",
                        help="Source directory with class folders")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for train/val split")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed
    )
