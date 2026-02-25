"""FastAPI application for rice leaf disease detection and fertilizer recommendation."""
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
import httpx  # Required for calling Ollama
from pydantic import BaseModel # Required for Soil Data validation

from app.schemas import PredictionResponse, HealthResponse
from app.inference import ModelInference


# Initialize FastAPI app
app = FastAPI(
    title="Rice Leaf Disease Detection & Agronomy API",
    description="API for detecting diseases in rice leaves and providing fertilizer advice using GenAI",
    version="2.0.0"
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
        print("API will start but image predictions will fail until model is loaded.")
        return
    
    print(f"Loading model from: {model_path}")
    try:
        model_inference = ModelInference(model_path, metadata_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")


@app.get("/")
async def root():
    """Serve the web interface."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": "Rice Leaf Disease & Fertilizer API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict_disease": "/predict",
            "recommend_fertilizer": "/recommend/fertilizer",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = model_inference is not None
    
    # You could also check if Ollama is reachable here
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get("http://ollama:11434")
            if resp.status_code == 200:
                ollama_status = "reachable"
    except:
        ollama_status = "unreachable"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message=f"Vision Model: {'Loaded' if model_loaded else 'Missing'}, AI Brain: {ollama_status}"
    )


# --- IMAGE PREDICTION ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image."""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Vision Model not loaded.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        result = model_inference.predict(image)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- NEW: Prompt Request Schema ---
class PromptRequest(BaseModel):
    prompt: str
    soil_data: dict = {} # Optional, for logging or metadata if needed
# --- NEW: FERTILIZER RECOMMENDATION ENDPOINT ---
@app.post("/recommend/fertilizer")
async def recommend_fertilizer(request: PromptRequest):
    """
    Receives a pre-constructed prompt from Laravel and forwards it to Ollama.
    """
    
    ai_advice = "AI Service Unavailable"
    
    # Use the prompt constructed in Laravel
    prompt = request.prompt
    try:
        # Call the sibling docker container "ollama"
        # We use a longer timeout (60s) because Generative AI takes a moment to "think"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json" # Ensure Ollama knows we want JSON
                }
            )
            if response.status_code == 200:
                # Get the 'response' field from Ollama's JSON output
                ai_advice = response.json().get('response', 'No response generated')
            else:
                print(f"Ollama Error: {response.text}")
                ai_advice = "Error: Ollama service returned an error."
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        ai_advice = "Could not connect to the AI Agronomist."
    # Return the raw JSON string from Ollama wrapped in a generic response key.
    # The Laravel service will parse the internal JSON structure.
    return {
        "response": ai_advice
    }


@app.get("/classes", response_model=Dict[str, List[str]])
async def get_classes():
    """Get list of disease classes."""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": model_inference.classes}

# Endpoint to use Ollama for different types of prompts like interacting with the GenAI for various advice, 
# not just fertilizer recommendations. 
# This keeps the API flexible and allows Laravel to construct any prompt it wants for different use cases.
class GenericPromptRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask(request: GenericPromptRequest):
    """
    A generic endpoint to handle any prompt from Laravel and forward it to Ollama.
    This can be used for various types of advice, not just fertilizer recommendations.
    """
    
    ai_response = "AI Service Unavailable"
    
    prompt = request.prompt
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                }
            )
            if response.status_code == 200:
                ai_response = response.json().get('response', 'No response generated')
            else:
                print(f"Ollama Error: {response.text}")
                ai_response = "Error: Ollama service returned an error."
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        ai_response = "Could not connect to the AI service."
    
    return {
        "response": ai_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)