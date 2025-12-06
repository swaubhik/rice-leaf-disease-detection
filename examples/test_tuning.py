"""
Example: Quick Hyperparameter Tuning Test
==========================================

This is a minimal example to test if Ray Tune is working correctly.
It runs a very short tuning session (2 trials, 3 epochs each) to verify setup.
"""

import os
import sys

# Ensure we're in the project directory
os.chdir('/home/stihub/ML/rice-disease-detection')

# Check if dependencies are installed
try:
    import ray
    from ray import tune
    import torch
    print("✓ Dependencies found")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Check if dataset exists
if not os.path.exists('data/train') or not os.path.exists('data/val'):
    print("✗ Dataset not found at data/train and data/val")
    print("\nPlease prepare dataset first:")
    print("  python prepare_dataset.py")
    sys.exit(1)

print("✓ Dataset found")

# Run a minimal tuning test
print("\n" + "="*80)
print("Running minimal hyperparameter tuning test...")
print("This will try 2 configurations with 3 epochs each")
print("="*80 + "\n")

os.system("""
python tune_hyperparameters.py \
    --data-dir data \
    --num-samples 2 \
    --max-epochs 3 \
    --grace-period 2 \
    --cpus-per-trial 2 \
    --gpus-per-trial 0.5 \
    --experiment-name test_tune
""")

print("\n" + "="*80)
print("Test complete! Check results at:")
print("  ray_results/test_tune/best_config.json")
print("="*80)
