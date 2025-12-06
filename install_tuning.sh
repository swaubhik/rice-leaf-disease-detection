#!/bin/bash
# Installation script for hyperparameter tuning dependencies

set -e  # Exit on error

echo "=================================================="
echo "Installing Ray Tune and Dependencies"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠ No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify Ray installation
echo "Verifying Ray Tune installation..."
python3 -c "import ray; from ray import tune; print('✓ Ray Tune version:', ray.__version__)" || {
    echo "✗ Ray Tune installation failed"
    exit 1
}
echo ""

# Verify other key dependencies
echo "Verifying other dependencies..."
python3 -c "import torch; print('✓ PyTorch version:', torch.__version__)"
python3 -c "import hyperopt; print('✓ HyperOpt installed')"
python3 -c "import pandas; print('✓ Pandas installed')"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('  GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

# Verify tuning script syntax
echo "Verifying tuning scripts..."
python3 -m py_compile tune_hyperparameters.py && echo "✓ tune_hyperparameters.py syntax OK"
python3 -m py_compile train_with_best_config.py && echo "✓ train_with_best_config.py syntax OK"
python3 -m py_compile analyze_tuning_results.py && echo "✓ analyze_tuning_results.py syntax OK"
echo ""

echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Prepare dataset: python prepare_dataset.py"
echo "  2. Run tuning:      python tune_hyperparameters.py --num-samples 5"
echo "  3. Train model:     python train_with_best_config.py --epochs 100"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART_TUNING.md (5-minute quick start)"
echo "  - HYPERPARAMETER_TUNING.md (comprehensive guide)"
echo ""
