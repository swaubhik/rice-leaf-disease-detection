# Rice Leaf Disease Detection System

This project implements a deep learning model for detecting diseases in rice leaves using PyTorch and provides a FastAPI-based REST API for inference.

## Project Structure
- `train.py`: Training script with transfer learning
- `dataset.py`: Dataset loader and preprocessing
- `models.py`: Model architecture definitions
- `export.py`: Model export to TorchScript/ONNX
- `app/`: FastAPI application
- `utils.py`: Utility functions
- `tests/`: Test cases

## Development Guidelines
- Use PyTorch for deep learning
- Follow type hints for all functions
- Keep API responses consistent with schemas
- Document all major functions
