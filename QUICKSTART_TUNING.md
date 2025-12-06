# Quick Start: Hyperparameter Tuning

This guide will get you started with hyperparameter tuning in under 5 minutes.

## Prerequisites

1. **Dataset prepared**: You should have `data/train` and `data/val` directories
   ```bash
   python prepare_dataset.py
   ```

2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Run Your First Tuning (5 minutes)

Start with a small test to ensure everything works:

```bash
python tune_hyperparameters.py \
    --data-dir data \
    --num-samples 5 \
    --max-epochs 10 \
    --gpus-per-trial 0.5
```

This will:
- Try 5 different hyperparameter combinations
- Train each for up to 10 epochs
- Use half a GPU per trial (allows 2 trials to run in parallel on 1 GPU)
- Save results to `ray_results/rice_disease_tune/`

**Expected time**: ~5-10 minutes (depends on your GPU)

## Step 2: Check Results

View the best configuration found:

```bash
cat ray_results/rice_disease_tune/best_config.json
```

Example output:
```json
{
  "config": {
    "model_name": "resnet50",
    "learning_rate": 0.0003,
    "batch_size": 32,
    "dropout": 0.4
  },
  "val_acc": 93.45,
  "val_loss": 0.1823
}
```

## Step 3: Train with Best Config

Use the best hyperparameters for full training:

```bash
python train_with_best_config.py --epochs 100
```

This will:
- Load the best config automatically
- Train for 100 epochs (longer than tuning)
- Save the final model to `models/`

## Step 4: Analyze Results (Optional)

Visualize the tuning results:

```bash
python analyze_tuning_results.py
```

This creates plots in `tune_analysis/` showing:
- Which hyperparameters had the biggest impact
- How trials improved over time
- Comparison of top configurations

## Next Steps

### For Better Results

Run a more comprehensive search:

```bash
python tune_hyperparameters.py \
    --data-dir data \
    --search-alg hyperopt \
    --num-samples 30 \
    --max-epochs 20 \
    --gpus-per-trial 0.5
```

### Different Search Algorithms

Try different search strategies:

**ASHA** (fastest, good for initial exploration):
```bash
python tune_hyperparameters.py --search-alg asha --num-samples 30
```

**HyperOpt** (smart Bayesian optimization):
```bash
python tune_hyperparameters.py --search-alg hyperopt --num-samples 25
```

**Population Based Training** (adapts during training):
```bash
python tune_hyperparameters.py --search-alg pbt --num-samples 15
```

### Customize Search Space

Edit `tune_hyperparameters.py` line ~180 to modify what gets tuned:

```python
config = {
    "model_name": tune.choice(["resnet50", "efficientnet_b0"]),
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64]),
    # Add or remove parameters here
}
```

## Common Issues

### Out of Memory
Reduce batch size or GPU allocation:
```bash
python tune_hyperparameters.py --gpus-per-trial 0.25 --num-samples 10
```

### Too Slow
Use fewer samples or shorter training:
```bash
python tune_hyperparameters.py --num-samples 10 --max-epochs 10
```

### Resume Interrupted Run
```bash
python tune_hyperparameters.py --resume --experiment-name rice_disease_tune
```

## Understanding the Output

While tuning runs, you'll see:

```
Trial status: 5 TERMINATED | 2 RUNNING | 3 PENDING
+------------------+------------+-------------+----------+
| Trial name       | val_acc    | train_loss  | epoch    |
+------------------+------------+-------------+----------+
| train_xyz_00001  | 94.23      | 0.167       | 15       |
| train_xyz_00002  | 91.45      | 0.223       | 12       |
+------------------+------------+-------------+----------+
```

- **TERMINATED**: Trial completed (or stopped early)
- **RUNNING**: Currently training
- **PENDING**: Waiting for resources
- **val_acc**: Validation accuracy (higher is better)

## Files Generated

After tuning completes:

```
ray_results/rice_disease_tune/
├── best_config.json          # Best hyperparameters found
├── all_results.csv            # All trial results
└── train_xyz_00001/          # Individual trial checkpoints
    ├── checkpoint.pt
    └── params.json
```

## Tips for Success

1. **Start small**: Test with 5 samples first
2. **Use GPU wisely**: Set `--gpus-per-trial 0.5` to run 2 trials per GPU
3. **Increase gradually**: If results look good, run more samples
4. **Compare algorithms**: Try ASHA first, then HyperOpt for refinement
5. **Monitor progress**: Watch validation accuracy trend upward

## Example Workflow

```bash
# 1. Quick test (5 min)
python tune_hyperparameters.py --num-samples 5 --max-epochs 10

# 2. Broader search (30 min)
python tune_hyperparameters.py --num-samples 20 --max-epochs 15

# 3. Analyze results
python analyze_tuning_results.py

# 4. Train final model (2 hours)
python train_with_best_config.py --epochs 100

# 5. Export for deployment
python export.py --checkpoint models/best_model.pth
```

## Getting Help

- Full documentation: [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)
- Ray Tune docs: https://docs.ray.io/en/latest/tune/
- Issues: Check `ray_results/rice_disease_tune/` for error logs

Happy tuning! 🚀
