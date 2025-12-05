"""Quick start script to verify dataset and environment."""
import os
import sys
from pathlib import Path

def check_dataset():
    """Check if dataset is properly split."""
    data_dir = Path("data")
    
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    if not data_dir.exists():
        print("❌ Dataset directory 'data/' not found!")
        print("   Run: python3 prepare_dataset.py")
        return False
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Missing train/ or val/ directories!")
        print("   Run: python3 prepare_dataset.py")
        return False
    
    # Count samples
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
    
    if len(train_classes) == 0 or len(val_classes) == 0:
        print("❌ No class directories found!")
        return False
    
    print(f"✅ Found {len(train_classes)} classes in training set")
    print(f"✅ Found {len(val_classes)} classes in validation set")
    
    total_train = 0
    total_val = 0
    
    print("\nClass distribution:")
    for class_dir in sorted(train_classes, key=lambda x: x.name):
        train_count = len(list(class_dir.glob("*")))
        val_count = len(list((val_dir / class_dir.name).glob("*")))
        total_train += train_count
        total_val += val_count
        print(f"  {class_dir.name:30s} - Train: {train_count:4d}, Val: {val_count:4d}")
    
    print(f"\n✅ Total training samples: {total_train}")
    print(f"✅ Total validation samples: {total_val}")
    
    return True

def check_dependencies():
    """Check if key dependencies are installed."""
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('fastapi', 'FastAPI'),
        ('PIL', 'Pillow'),
    ]
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install -r requirements.txt")
        return False
    
    return True

def show_next_steps():
    """Show next steps."""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    print("\n1. Train the model:")
    print("   python3 train.py --data-dir data --epochs 50 --batch-size 32")
    
    print("\n2. Export the trained model:")
    print("   python3 export.py --model-dir models --model-file best_model.pth")
    
    print("\n3. Start the API server:")
    print("   python3 -m uvicorn app.main:app --reload")
    
    print("\n4. Test the API:")
    print("   Open http://localhost:8001/docs in your browser")
    print("   Or use: curl -X POST http://localhost:8001/predict -F 'file=@image.jpg'")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("\n🌾 Rice Leaf Disease Detection System - Setup Verification\n")
    
    dataset_ok = check_dataset()
    deps_ok = check_dependencies()
    
    if dataset_ok and deps_ok:
        print("\n✅ All checks passed! System is ready.")
        show_next_steps()
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
