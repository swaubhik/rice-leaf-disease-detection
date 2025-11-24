# 🚀 Optimized Training for NVIDIA A6000

With your NVIDIA A6000 (48GB VRAM), you can train much faster and with better hyperparameters!

## Recommended Training Commands

### Fast Training (ResNet50) - ~10-15 minutes ⚡ RECOMMENDED
```bash
python3 train.py \
  --data-dir data \
  --model-name resnet50 \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --freeze-backbone-epochs 5 \
  --num-workers 8
```

### Best Accuracy (EfficientNet-B3) - ~30-40 minutes
```bash
python3 train.py \
  --data-dir data \
  --model-name efficientnet_b3 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --freeze-backbone-epochs 10 \
  --num-workers 8
```

### Maximum Performance (EfficientNet-B3 with larger images)
```bash
python3 train.py \
  --data-dir data \
  --model-name efficientnet_b3 \
  --image-size 299 \
  --epochs 100 \
  --batch-size 48 \
  --learning-rate 0.0003 \
  --freeze-backbone-epochs 10 \
  --num-workers 8
```

### Ultra-Fast Testing (5 minutes)
```bash
python3 train.py \
  --data-dir data \
  --model-name resnet50 \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 8
```

## Performance Tips for A6000

1. **Large Batch Sizes**: Use 64-256 depending on model
2. **More Workers**: Set `--num-workers 8` or higher
3. **Mixed Precision**: Enable AMP for 2-3x speedup (will add this)
4. **Larger Images**: Use `--image-size 299` or `384` for better accuracy

## Monitor GPU Usage

```bash
# Install nvitop (you already have it!)
nvitop

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

## Expected Training Times on A6000

| Model | Batch Size | Image Size | Time per Epoch | Total (50 epochs) |
|-------|-----------|------------|----------------|-------------------|
| ResNet50 | 128 | 224 | ~12 sec | ~10 min |
| ResNet101 | 64 | 224 | ~18 sec | ~15 min |
| EfficientNet-B0 | 128 | 224 | ~15 sec | ~12 min |
| EfficientNet-B3 | 64 | 299 | ~35 sec | ~30 min |

## Training with Multiple GPUs (if you have more)

```bash
# Use DataParallel for multi-GPU
python3 train.py --data-dir data --batch-size 256
```

Enjoy the speed! 🚀
