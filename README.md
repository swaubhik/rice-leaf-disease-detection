# Rice Leaf Disease Detection System

A complete deep learning system for detecting diseases in rice leaves using PyTorch and FastAPI. The system includes training scripts with transfer learning, model export capabilities, and a production-ready REST API for inference.

## Features

- **Transfer Learning**: Uses pre-trained models (ResNet50, ResNet101, EfficientNet) for better accuracy
- **Data Augmentation**: Built-in augmentation for robust training
- **Model Export**: Export to TorchScript and ONNX for production deployment
- **REST API**: FastAPI-based API with automatic documentation
- **Docker Support**: Containerized deployment ready
- **Comprehensive Testing**: Unit tests for API endpoints

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── inference.py     # Inference utilities
│   └── schemas.py       # Pydantic schemas
├── tests/
│   ├── __init__.py
│   └── test_api.py      # API tests
├── dataset.py           # Dataset loader
├── models.py            # Model architectures
├── train.py             # Training script
├── export.py            # Model export script
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── README.md            # This file
```

## Installation

### Local Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ML
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the dataset in `Augmented Images/` directory with 8 disease classes:
- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf scald
- Narrow Brown Leaf Spot
- Rice Hispa
- Sheath Blight

### Split Dataset into Train/Val

Run the preparation script to split the dataset:

```bash
python prepare_dataset.py --source-dir "Augmented Images" --output-dir data --val-split 0.2
```

This will create:
```
data/
├── train/
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   ├── Healthy Rice Leaf/
│   └── ... (all 8 classes)
└── val/
    ├── Bacterial Leaf Blight/
    ├── Brown Spot/
    └── ... (all 8 classes)
```

**Options:**
- `--source-dir`: Source directory (default: `Augmented Images`)
- `--output-dir`: Output directory (default: `data`)
- `--val-split`: Validation ratio, e.g., 0.2 = 20% (default: `0.2`)
- `--seed`: Random seed for reproducibility (default: `42`)

## Training

### Prepare Dataset First

Before training, split the dataset:

```bash
python prepare_dataset.py
```

### Basic Training

Train with default settings (ResNet50):

```bash
python train.py --data-dir data --epochs 50 --batch-size 32
```

### Advanced Training Options

```bash
python train.py \
  --data-dir data \
  --model-name efficientnet_b3 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --image-size 224 \
  --freeze-backbone-epochs 5 \
  --checkpoint-dir checkpoints \
  --output-dir models
```

### Training Parameters

- `--data-dir`: Path to dataset directory (default: `data`)
- `--model-name`: Model architecture - `resnet50`, `resnet101`, `efficientnet_b0`, `efficientnet_b3` (default: `resnet50`)
- `--epochs`: Number of training epochs (default: `50`)
- `--batch-size`: Batch size (default: `32`)
- `--learning-rate`: Initial learning rate (default: `0.001`)
- `--image-size`: Input image size (default: `224`)
- `--freeze-backbone-epochs`: Freeze backbone for N epochs (default: `0`)
- `--checkpoint-dir`: Directory for checkpoints (default: `checkpoints`)
- `--output-dir`: Directory for final models (default: `models`)
- `--resume`: Path to checkpoint to resume training

### Resume Training

```bash
python train.py --resume checkpoints/best_model_checkpoint.pth
```

## Model Export

Export the trained model to TorchScript and ONNX formats:

```bash
python export.py \
  --model-dir models \
  --model-file best_model.pth \
  --model-name resnet50 \
  --output-dir exported_models
```

This creates:
- `exported_models/model.pt` (TorchScript)
- `exported_models/model.onnx` (ONNX)
- `exported_models/model_metadata.json` (Metadata)

## API Usage

### Start the API Server

**Local Development**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Production**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

The API will be available at `http://localhost:8000`

- **Health Check**: `GET /health`
- **Predict Disease**: `POST /predict`
- **Get Classes**: `GET /classes`
- **API Documentation**: `GET /docs` (Swagger UI)
- **Alternative Docs**: `GET /redoc` (ReDoc)

### Example API Calls

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Predict Disease**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/rice_leaf.jpg"
```

**Get Classes**:
```bash
curl http://localhost:8000/classes
```

### Python Client Example

```python
import requests

# Predict disease
url = "http://localhost:8000/predict"
files = {"file": open("rice_leaf.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top predictions: {result['top_predictions']}")
```

## Docker Deployment

### Build and Run with Docker

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
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Model Performance Tips

1. **Data Augmentation**: Adjust augmentation in `dataset.py` for your specific use case
2. **Learning Rate**: Use learning rate schedulers for better convergence
3. **Freeze/Unfreeze**: Freeze backbone initially, then fine-tune for best results
4. **Batch Size**: Larger batch sizes generally improve training stability
5. **Image Size**: Higher resolution (e.g., 299 for EfficientNet) may improve accuracy

## Production Considerations

1. **Model Optimization**: Use ONNX Runtime for faster inference
2. **Caching**: Implement model caching for multi-worker setups
3. **Monitoring**: Add logging and metrics collection
4. **Rate Limiting**: Implement rate limiting for API endpoints
5. **Input Validation**: Validate image size and format before processing
6. **Error Handling**: Comprehensive error handling for production robustness

## Environment Variables

- `MODEL_PATH`: Path to model file (default: `exported_models/model.pt`)
- `METADATA_PATH`: Path to metadata JSON (default: `exported_models/model_metadata.json`)

## Common Issues

### Model Not Found
Ensure you've exported the model before starting the API:
```bash
python export.py
```

### CUDA Out of Memory
Reduce batch size or use a smaller model:
```bash
python train.py --batch-size 16 --model-name resnet50
```

### Slow Inference
Use ONNX Runtime for faster CPU inference:
```bash
export MODEL_PATH=exported_models/model.onnx
uvicorn app.main:app
```

## License

MIT License - feel free to use this project for your own purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTorch team for the excellent framework
- FastAPI for the modern API framework
- Pre-trained models from torchvision

## Contact

For questions or issues, please open an issue on GitHub.
