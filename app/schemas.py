"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict


class PredictionResponse(BaseModel):
    """Response model for disease prediction."""
    predicted_class: str = Field(..., description="Predicted disease class")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    top_predictions: List[Dict[str, float]] = Field(
        ..., 
        description="Top N predictions with class names and confidence scores"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "bacterial_leaf_blight",
                "confidence": 0.92,
                "top_predictions": [
                    {"bacterial_leaf_blight": 0.92},
                    {"brown_spot": 0.05},
                    {"leaf_smut": 0.02}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    message: str = Field(..., description="Status message")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "message": "API is running"
            }
        }
