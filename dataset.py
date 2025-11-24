"""Dataset loader for rice leaf disease detection."""
import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class RiceLeafDataset(Dataset):
    """Rice Leaf Disease Dataset.
    
    Expected directory structure:
        data/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img1.jpg
            val/
                class1/
                class2/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory containing train/val folders
            split: 'train' or 'val'
            transform: Optional transforms to apply
            target_size: Target image size (height, width)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size
        
        # Get data directory
        self.data_dir = self.root_dir / split
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Get class names from subdirectories
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        self._load_samples()
        
        # Set transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _load_samples(self):
        """Load all image paths and their labels."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms for train/val split."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders.
    
    Args:
        data_dir: Root directory containing train/val folders
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Create datasets
    train_dataset = RiceLeafDataset(
        root_dir=data_dir,
        split="train",
        target_size=target_size
    )
    
    val_dataset = RiceLeafDataset(
        root_dir=data_dir,
        split="val",
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    
    if os.path.exists(data_dir):
        train_loader, val_loader, classes = create_data_loaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=2
        )
        
        print(f"Classes: {classes}")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    else:
        print(f"Data directory '{data_dir}' not found.")
        print("Please run: python prepare_dataset.py")
        print("This will split 'Augmented Images/' into train/val sets.")
