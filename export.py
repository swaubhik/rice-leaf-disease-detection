"""Export trained model to TorchScript and ONNX formats."""
import argparse
import json
import os
import torch
import torch.nn as nn
from pathlib import Path

from models import create_model


def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    example_input: torch.Tensor,
    use_trace: bool = True
) -> str:
    """Export model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        save_path: Path to save TorchScript model
        example_input: Example input tensor for tracing
        use_trace: If True, use torch.jit.trace; else use torch.jit.script
        
    Returns:
        Path to saved TorchScript model
    """
    model.eval()
    
    if use_trace:
        traced_model = torch.jit.trace(model, example_input)
    else:
        traced_model = torch.jit.script(model)
    
    traced_model.save(save_path)
    print(f"TorchScript model saved to: {save_path}")
    
    # Verify the exported model
    loaded_model = torch.jit.load(save_path)
    with torch.no_grad():
        original_output = model(example_input)
        loaded_output = loaded_model(example_input)
        max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
        print(f"Max difference between original and loaded model: {max_diff:.6f}")
    
    return save_path


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    example_input: torch.Tensor,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    opset_version: int = 14
) -> str:
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        example_input: Example input tensor
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes for variable batch size
        opset_version: ONNX opset version
        
    Returns:
        Path to saved ONNX model
    """
    model.eval()
    
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        example_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model saved to: {save_path}")
    
    # Verify the exported model
    try:
        import onnx
        import onnxruntime as ort
        
        # Check ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Test inference
        ort_session = ort.InferenceSession(save_path)
        ort_inputs = {input_names[0]: example_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        with torch.no_grad():
            torch_output = model(example_input).cpu().numpy()
        
        max_diff = abs(torch_output - ort_outputs[0]).max()
        print(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
    except ImportError:
        print("Warning: onnx or onnxruntime not installed. Skipping verification.")
    
    return save_path


def main(args):
    """Main export function."""
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load class names
    class_names_path = os.path.join(args.model_dir, "class_names.json")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names not found at {class_names_path}")
    
    with open(class_names_path, 'r') as f:
        classes = json.load(f)
    
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=False,
        device=device
    )
    
    # Load trained weights
    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Create example input
    example_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export to TorchScript
    if args.export_torchscript:
        torchscript_path = os.path.join(args.output_dir, "model.pt")
        print(f"\nExporting to TorchScript...")
        export_to_torchscript(model, torchscript_path, example_input, use_trace=True)
    
    # Export to ONNX
    if args.export_onnx:
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        print(f"\nExporting to ONNX...")
        export_to_onnx(
            model,
            onnx_path,
            example_input,
            opset_version=args.onnx_opset
        )
    
    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "classes": classes,
        "image_size": args.image_size,
        "input_shape": [1, 3, args.image_size, args.image_size],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    
    metadata_path = os.path.join(args.output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nModel metadata saved to: {metadata_path}")
    
    print("\nExport complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model to production formats")
    
    # Model arguments
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained model")
    parser.add_argument("--model-file", type=str, default="best_model.pth",
                        help="Model file name")
    parser.add_argument("--model-name", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3"],
                        help="Model architecture")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    
    # Export options
    parser.add_argument("--export-torchscript", action="store_true", default=True,
                        help="Export to TorchScript format")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export to ONNX format")
    parser.add_argument("--onnx-opset", type=int, default=14,
                        help="ONNX opset version")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="exported_models",
                        help="Directory to save exported models")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    main(args)
