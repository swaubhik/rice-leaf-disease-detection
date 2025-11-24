"""Model inference utilities."""
import json
import os
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class ModelInference:
    """Handle model loading and inference."""
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str = None,
        device: str = None
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to exported model (.pt or .onnx)
            metadata_path: Path to model metadata JSON
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Use defaults if metadata not found
            self.metadata = {
                "image_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "classes": []
            }
        
        self.classes = self.metadata.get("classes", [])
        self.image_size = self.metadata.get("image_size", 224)
        
        # Load model
        self.model = self._load_model()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.metadata.get("mean", [0.485, 0.456, 0.406]),
                std=self.metadata.get("std", [0.229, 0.224, 0.225])
            )
        ])
    
    def _load_model(self):
        """Load the model based on file extension."""
        if self.model_path.endswith('.pt'):
            # Load TorchScript model
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        elif self.model_path.endswith('.onnx'):
            # Load ONNX model
            import onnxruntime as ort
            model = ort.InferenceSession(self.model_path)
            return model
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict:
        """Make prediction on image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        if self.model_path.endswith('.pt'):
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Get probabilities
            probabilities = F.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
        elif self.model_path.endswith('.onnx'):
            # ONNX inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            input_data = input_tensor.numpy()
            output = self.model.run([output_name], {input_name: input_data})[0]
            
            # Get probabilities
            output_tensor = torch.from_numpy(output)
            probabilities = F.softmax(output_tensor, dim=1).numpy()[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = [
            {self.classes[idx]: float(probabilities[idx])}
            for idx in top_indices
        ]
        
        # Get top prediction
        predicted_idx = top_indices[0]
        predicted_class = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
    
    def predict_batch(self, images: List[Image.Image], top_k: int = 3) -> List[Dict]:
        """Make predictions on batch of images.
        
        Args:
            images: List of PIL Images
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(image, top_k) for image in images]
