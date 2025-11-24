"""Tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "message" in data
    assert data["status"] in ["healthy", "degraded"]


def test_predict_no_file():
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_file():
    """Test predict endpoint with invalid file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_get_classes():
    """Test get classes endpoint."""
    response = client.get("/classes")
    # Will return 503 if model not loaded, which is acceptable
    assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
