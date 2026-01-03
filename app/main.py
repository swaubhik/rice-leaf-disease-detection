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

# --- NEW: Soil Data Schema ---
class SoilData(BaseModel):
    n: int
    p: int
    k: int
    ph: float
    ec: float = 0.0 # Default to 0 if not provided
    soilTemp: float = 25.0 # Default to 25°C if not provided
    soilMoisture: float = 0.0 # Default to 0 if not provided
    crop : str = "rice" # Default crop is rice

# --- NEW: FERTILIZER RECOMMENDATION ENDPOINT ---
@app.post("/recommend/fertilizer")
async def recommend_fertilizer(data: SoilData):
    """
    1. Calculates NPK deficit using standard math (Safety Layer).
    2. Asks Local LLM (Ollama) for advice and context (Intelligence Layer).
    """
    
    # 1. Deterministic Math Rule (Safety Layer)
    # Simple logic: Target N for Rice is approx 120 kg/ha.
    # Urea has 46% N. So 1 kg N needs ~2.2 kg Urea.
    needed_n = max(0, 120 - data.n)
    needed_urea = needed_n * 2.2 

    # 2. AI Context Layer (Ollama on A6000)
    # This prompt tells Llama 3 exactly what to do.
    prompt = f"""
    You are an expert agronomist in Assam, India.
    Analyze this soil sample for growing {data.crop}.
    Provide concise fertilizer advice based on the following data:
    - Nitrogen: {data.n} mg/kg (Calculated deficit: {needed_n} mg/kg)
    - Phosphorus: {data.p} mg/kg
    - Potassium: {data.k} mg/kg
    - pH: {data.ph}
    - EC: {data.ec}
    - Soil Temperature: {data.soilTemp} °C
    - Soil Moisture: {data.soilMoisture} %

    The math suggests applying {needed_urea:.1f} kg/ha of Urea. 
    
    TASK:
    1. Confirm if this Urea dosage is safe based on the pH.
    2. Suggest 1 organic alternative available in India.
    3. If pH is acidic (< 6.0) or alkaline (> 7.5), suggest a correction (Lime/Gypsum).
    4. Keep the response concise (under 4 sentences).
    """

    ai_advice = "AI Service Unavailable"

    try:
        # Call the sibling docker container "ollama"
        # We use a longer timeout (45s) because Generative AI takes a moment to "think"
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                ai_advice = response.json().get('response', 'No response generated')
            else:
                print(f"Ollama Error: {response.text}")

    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        ai_advice = "Could not connect to the AI Agronomist. Please follow standard NPK charts."

    return {
        "soil_status": {
            "nitrogen_level": "Low" if data.n < 120 else "Adequate",
            "ph_status": "Acidic" if data.ph < 6.0 else ("Alkaline" if data.ph > 7.5 else "Neutral")
        },
        "calculated_dosage": {
            "urea_kg_per_ha": round(needed_urea, 2),
            "dap_kg_per_ha": 0 # Placeholder for DAP logic
        },
        "agronomist_advice": ai_advice
    }


@app.get("/classes", response_model=Dict[str, List[str]])
async def get_classes():
    """Get list of disease classes."""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": model_inference.classes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)