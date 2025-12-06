# Hyperparameter Tuning Guide

This guide explains how to use Ray Tune to automatically find optimal hyperparameters for the rice leaf disease detection model.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (ASHA Scheduler)

Run hyperparameter tuning with the default ASHA (Asynchronous Successive Halving) scheduler:

```bash
python tune_hyperparameters.py --data-dir data --num-samples 20
```

This will:
- Try 20 different hyperparameter configurations
- Train each for up to 20 epochs
- Automatically stop poorly performing trials early
- Save results to `ray_results/rice_disease_tune/`

### Advanced Search Algorithms

#### 1. Population Based Training (PBT)
Best for: Dynamically adjusting hyperparameters during training

```bash
python tune_hyperparameters.py \
    --data-dir data \
    --search-alg pbt \
    --num-samples 10 \
    --max-epochs 30
```

#### 2. HyperOpt (Tree-structured Parzen Estimator)
Best for: Efficient Bayesian optimization with many hyperparameters

```bash
python tune_hyperparameters.py \
    --data-dir data \
    --search-alg hyperopt \
    --num-samples 30 \
    --max-epochs 20
```

#### 3. Bayesian Optimization
Best for: Small number of expensive evaluations

```bash
python tune_hyperparameters.py \
    --data-dir data \
    --search-alg bayesopt \
    --num-samples 15 \
    --max-epochs 25
```

## Hyperparameters Being Tuned

The script automatically searches over:

### Model Architecture
- **model_name**: resnet50, efficientnet_b0
- **dropout**: 0.2 to 0.6
- **image_size**: 224, 256, 288

### Optimizer
- **optimizer**: adam, adamw, sgd
- **learning_rate**: 1e-5 to 1e-2 (log scale)
- **weight_decay**: 1e-6 to 1e-3 (log scale)
- **momentum**: 0.85 to 0.95 (for SGD)

### Training Settings
- **batch_size**: 16, 24, 32, 48, 64
- **scheduler**: cosine, step, reduce_on_plateau
- **label_smoothing**: 0.0 to 0.2

## Resource Configuration

### GPU Settings

For single GPU with multiple trials:
```bash
python tune_hyperparameters.py \
    --gpus-per-trial 0.5 \
    --num-samples 20
```

For multiple GPUs (e.g., 4 GPUs):
```bash
python tune_hyperparameters.py \
    --gpus-per-trial 1 \
    --num-samples 20
```

### CPU Settings

Adjust CPUs per trial:
```bash
python tune_hyperparameters.py \
    --cpus-per-trial 8 \
    --num-samples 10
```

## Resuming Interrupted Runs

If tuning is interrupted, resume from checkpoint:

```bash
python tune_hyperparameters.py \
    --resume \
    --experiment-name rice_disease_tune
```

## Understanding Results

### Best Configuration

The best hyperparameters are saved to:
```
ray_results/rice_disease_tune/best_config.json
```

Example output:
```json
{
  "config": {
    "model_name": "efficientnet_b0",
    "dropout": 0.35,
    "image_size": 256,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "scheduler": "cosine",
    "label_smoothing": 0.1
  },
  "val_acc": 95.67,
  "val_loss": 0.1234
}
```

### All Results

All trial results are saved to:
```
ray_results/rice_disease_tune/all_results.csv
```

You can analyze this CSV to understand which hyperparameters had the most impact.

## Using Best Configuration

Once you have the best config, train a full model:

```bash
# Extract values from best_config.json and use them
python train.py \
    --data-dir data \
    --model-name efficientnet_b0 \
    --dropout 0.35 \
    --image-size 256 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --batch-size 32 \
    --epochs 100
```

## Performance Optimization Tips

### 1. Start Small
Begin with fewer samples to test the setup:
```bash
python tune_hyperparameters.py --num-samples 5 --max-epochs 10
```

### 2. Increase Grace Period
Allow more epochs before early stopping for complex models:
```bash
python tune_hyperparameters.py --grace-period 10 --max-epochs 30
```

### 3. Custom Experiment Name
Organize multiple tuning runs:
```bash
python tune_hyperparameters.py \
    --experiment-name resnet_tune_v1 \
    --search-alg asha
```

### 4. Limit Search Space
Edit `tune_hyperparameters.py` to focus on specific hyperparameters:

```python
config = {
    "model_name": tune.choice(["resnet50"]),  # Fix model
    "learning_rate": tune.loguniform(1e-4, 1e-2),  # Narrow range
    "batch_size": tune.choice([32, 64]),  # Fewer options
    # ...
}
```

## Monitoring Progress

Ray Tune provides a web UI. Install and run:

```bash
pip install ray[default]
tensorboard --logdir ray_results/rice_disease_tune
```

Or monitor in terminal:
- Training progress updates every 30 seconds
- Shows best trial so far
- Displays resource utilization

## Troubleshooting

### Out of Memory
Reduce batch size or GPU allocation:
```bash
python tune_hyperparameters.py --gpus-per-trial 0.25
```

### Too Slow
Increase GPU allocation or reduce num_samples:
```bash
python tune_hyperparameters.py \
    --gpus-per-trial 1 \
    --num-samples 10
```

### Ray Cluster
For distributed tuning across multiple machines:
```bash
# On head node
ray start --head

# On worker nodes
ray start --address=<head-node-ip>:6379

# Run tuning
python tune_hyperparameters.py --num-samples 50
```

## Comparison of Search Algorithms

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **ASHA** | Quick exploration | Fast, efficient early stopping | May miss global optimum |
| **PBT** | Dynamic adjustment | Adapts during training | Requires more resources |
| **HyperOpt** | Complex search spaces | Smart sampling, proven results | Slower than ASHA |
| **BayesOpt** | Few expensive trials | Very efficient sampling | Limited to smaller search spaces |

## Example Workflow

1. **Initial exploration** (ASHA with many trials):
```bash
python tune_hyperparameters.py --search-alg asha --num-samples 30 --max-epochs 15
```

2. **Refine top configs** (HyperOpt with narrow range):
Edit search space based on step 1 results, then:
```bash
python tune_hyperparameters.py --search-alg hyperopt --num-samples 20 --max-epochs 30
```

3. **Final training** (use best config):
```bash
python train.py <args-from-best-config> --epochs 100
```

## Additional Resources

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [Hyperparameter Tuning Best Practices](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html)
- [Search Algorithm Comparison](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html)
