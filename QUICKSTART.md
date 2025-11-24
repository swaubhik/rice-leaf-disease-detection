# 🌾 Rice Leaf Disease Detection System - READY TO USE!

## ✅ Dataset Status

Your dataset has been successfully prepared:

- **Training samples**: 4,153 images (80%)
- **Validation samples**: 1,035 images (20%)
- **Total images**: 5,188
- **Number of classes**: 8

### Class Distribution

| Class Name                  | Training | Validation | Total |
|----------------------------|----------|------------|-------|
| Bacterial Leaf Blight      | 429      | 107        | 536   |
| Brown Spot                 | 648      | 162        | 810   |
| Healthy Rice Leaf          | 409      | 102        | 511   |
| Leaf Blast                 | 744      | 185        | 929   |
| Leaf scald                 | 448      | 112        | 560   |
| Narrow Brown Leaf Spot     | 283      | 70         | 353   |
| Rice Hispa                 | 532      | 132        | 664   |
| Sheath Blight              | 660      | 165        | 825   |

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer pip3:
```bash
pip3 install -r requirements.txt
```

### Step 2: Verify Setup

```bash
python3 verify_setup.py
```

### Step 3: Train the Model

**Quick training (for testing)**:
```bash
python3 train.py --data-dir data --epochs 10 --batch-size 16
```

**Full training (recommended)**:
```bash
python3 train.py \
  --data-dir data \
  --model-name resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --freeze-backbone-epochs 5
```

**For better accuracy (EfficientNet)**:
```bash
python3 train.py \
  --data-dir data \
  --model-name efficientnet_b3 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.0005
```

Training outputs:
- Checkpoints saved to: `checkpoints/`
- Best model saved to: `models/best_model.pth`
- Training plot saved to: `models/training_history.png`
- Class names saved to: `models/class_names.json`

### Step 4: Export Model for Production

```bash
python3 export.py \
  --model-dir models \
  --model-file best_model.pth \
  --model-name resnet50 \
  --output-dir exported_models
```

This creates:
- `exported_models/model.pt` (TorchScript - recommended)
- `exported_models/model.onnx` (ONNX)
- `exported_models/model_metadata.json` (metadata)

### Step 5: Start the API

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API endpoint: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Step 6: Test the API

**Using the browser**:
1. Open http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Upload an image
5. Click "Execute"

**Using curl**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/rice_leaf.jpg"
```

**Using Python**:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("rice_leaf.jpg", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t rice-disease-api .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/exported_models:/app/exported_models:ro \
  rice-disease-api
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## 📊 Expected Training Results

With the current dataset (5,188 images, 8 classes), you can expect:

- **Training time**: 
  - ResNet50: ~15-30 minutes per epoch (GPU) / ~2-3 hours (CPU)
  - EfficientNet-B3: ~30-45 minutes per epoch (GPU) / ~4-6 hours (CPU)

- **Expected accuracy**: 
  - After 10 epochs: 70-80%
  - After 50 epochs: 85-95%
  - After 100 epochs with fine-tuning: 95%+

- **Model size**:
  - ResNet50: ~100 MB
  - EfficientNet-B3: ~50 MB

## 🎯 Inference Speed

On a typical CPU:
- **TorchScript (.pt)**: ~100-200 ms per image
- **ONNX (.onnx)**: ~50-100 ms per image (faster!)

On GPU:
- **TorchScript (.pt)**: ~10-20 ms per image
- **ONNX (.onnx)**: ~5-10 ms per image

## 📁 Project Files

### Core Files
- `prepare_dataset.py` - Split dataset into train/val
- `dataset.py` - Dataset loader with augmentation
- `models.py` - Model architectures
- `train.py` - Training script
- `export.py` - Export to TorchScript/ONNX
- `utils.py` - Training utilities

### API Files
- `app/main.py` - FastAPI application
- `app/inference.py` - Inference engine
- `app/schemas.py` - API schemas

### Configuration
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Docker orchestration

### Testing
- `tests/test_api.py` - API tests
- `verify_setup.py` - Setup verification

## 🔧 Troubleshooting

### Out of Memory during training
```bash
# Reduce batch size
python3 train.py --batch-size 8
```

### Slow training
```bash
# Reduce image size
python3 train.py --image-size 128

# Or use a smaller model
python3 train.py --model-name resnet50
```

### API returns "Model not loaded"
```bash
# Make sure you exported the model first
python3 export.py

# Check the exported_models directory exists
ls exported_models/
```

## 📚 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Transfer Learning Guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 🎉 You're All Set!

Your rice leaf disease detection system is ready to use. The dataset is prepared, and you can now:

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Train the model: `python3 train.py --data-dir data --epochs 50`
3. ✅ Export the model: `python3 export.py`
4. ✅ Start the API: `python3 -m uvicorn app.main:app`
5. ✅ Make predictions via API at http://localhost:8000/docs

Happy coding! 🚀
