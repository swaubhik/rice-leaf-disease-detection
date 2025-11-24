"""FastAPI application for rice leaf disease detection."""
import os
import json
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

from app.schemas import PredictionResponse, HealthResponse
from app.inference import ModelInference


# Initialize FastAPI app
app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="API for detecting diseases in rice leaves using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global model inference object
model_inference = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model_inference
    
    # Get model path from environment or use default
    model_path = os.getenv("MODEL_PATH", "exported_models/model.pt")
    metadata_path = os.getenv("METADATA_PATH", "exported_models/model_metadata.json")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("API will start but predictions will fail until model is loaded.")
        return
    
    print(f"Loading model from: {model_path}")
    model_inference = ModelInference(model_path, metadata_path)
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Serve the web interface."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": "Rice Leaf Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "web": "/"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = model_inference is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message="API is running" if model_loaded else "Model not loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction response with class, confidence, and top predictions
    """
    # Check if model is loaded
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Make prediction
        result = model_inference.predict(image)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/classes", response_model=Dict[str, List[str]])
async def get_classes():
    """Get list of disease classes."""
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {"classes": model_inference.classes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
