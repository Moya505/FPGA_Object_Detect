# quantize_yolov11.py - Complete quantization script following AMD Vitis AI documentation
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import argparse

# Step 1: Import the vai_q_pytorch module
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

# Import your model
#from model import YOLOv11
from model_vitis_compatible import YOLOv11VitisCompatible as YOLOv11

class CalibrationDataset(Dataset):
    """
    Calibration dataset for quantization
    Should contain 100-1000 representative images
    """
    def __init__(self, data_dir, transform=None, max_images=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        
        # Supported image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # Collect image paths
        for ext in extensions:
            pattern = os.path.join(data_dir, '**', ext)
            self.images.extend(glob.glob(pattern, recursive=True))
        
        # Shuffle and limit to max_images
        import random
        random.shuffle(self.images)
        self.images = self.images[:max_images]
        
        print(f"Calibration dataset: {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0  # Return dummy label
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = Image.new('RGB', (640, 640), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

def create_data_loader(data_dir, batch_size=1, input_size=640, max_images=1000):
    """
    Create data loader for calibration
    """
    # YOLOv11 preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),  # Converts to [0,1] range
        # Note: YOLOv11 expects [0,1] normalized images, not ImageNet normalization
    ])
    
    dataset = CalibrationDataset(data_dir, transform=transform, max_images=max_images)
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return loader

def evaluate(model, data_loader, device='cpu'):
    """
    Evaluation function for the quantized model
    This is a simplified evaluation - you can extend it based on your needs
    """
    model.eval()
    total_batches = 0
    successful_forwards = 0
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            try:
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Check if outputs are valid
                if isinstance(outputs, (list, tuple)):
                    all_valid = all(torch.isfinite(out).all() for out in outputs)
                else:
                    all_valid = torch.isfinite(outputs).all()
                
                if all_valid:
                    successful_forwards += 1
                
                total_batches += 1
                
                # Limit evaluation for calibration (process subset)
                if i >= 100:  # Process first 100 batches for calibration
                    break
                    
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1} batches...")
                    
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                total_batches += 1
    
    success_rate = successful_forwards / total_batches if total_batches > 0 else 0
    print(f"Evaluation complete: {successful_forwards}/{total_batches} successful forwards ({success_rate:.2%})")
    
    return success_rate, 0, 0  # Return dummy values for compatibility

def quantize_model(model_path, calibration_data_dir, quant_mode='calib', deploy=False, device='cpu'):
    """
    Main quantization function following AMD Vitis AI documentation
    """
    print(f"=== YOLOv11 Quantization (mode: {quant_mode}) ===")
    
    # Load the model
    print("1. Loading model...")
    model = YOLOv11(nc=11)  # Your model has 11 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Create calibration data loader
    print("2. Loading calibration data...")
    try:
        val_loader = create_data_loader(
            calibration_data_dir, 
            batch_size=1, 
            input_size=640, 
            max_images=500  # Use 500 images for calibration
        )
        print(f"   ✓ Calibration data loaded: {len(val_loader.dataset)} images")
    except Exception as e:
        print(f"   ✗ Error loading calibration data: {e}")
        return False
    
    # Step 2: Generate a quantizer with quantization needed input and get the converted model
    print("3. Creating quantizer...")
    input_tensor = torch.randn([1, 3, 640, 640]).to(device)
    
    try:
        quantizer = torch_quantizer(quant_mode, model, (input_tensor,))
        quant_model = quantizer.quant_model
        print("   ✓ Quantizer created successfully")
    except Exception as e:
        print(f"   ✗ Error creating quantizer: {e}")
        return False
    
    # Step 3: Forward a neural network with the converted model
    print("4. Running evaluation with quantized model...")
    try:
        acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, device)
        print(f"   ✓ Evaluation completed - Success rate: {acc1_gen:.2%}")
    except Exception as e:
        print(f"   ✗ Error during evaluation: {e}")
        return False
    
    # Step 4: Output the quantization result and deploy the model
    print("5. Exporting quantization results...")
    
    if quant_mode == 'calib':
        try:
            quantizer.export_quant_config()
            print("   ✓ Quantization config exported")
            print("   ℹ️  Next step: Run with quant_mode='test' to generate deployable model")
        except Exception as e:
            print(f"   ✗ Error exporting quant config: {e}")
            return False
    
    if deploy:
        try:
            print("   Exporting deployment models...")
            
            # Export TorchScript model
            quantizer.export_torch_script()
            print("   ✓ TorchScript model exported")
            
            # Export ONNX model
            quantizer.export_onnx_model()
            print("   ✓ ONNX model exported")
            
            # Export XMODEL for FPGA deployment
            quantizer.export_xmodel(deploy_check=False)
            print("   ✓ XMODEL exported for FPGA deployment")
            
        except Exception as e:
            print(f"   ✗ Error exporting deployment models: {e}")
            return False
    
    print("✅ Quantization process completed successfully!")
    return True

def compile_for_zcu104(xmodel_path, output_dir="compiled_model"):
    """
    Compile the quantized model for ZCU104 FPGA
    """
    print("=== Compiling for ZCU104 FPGA ===")
    
    import subprocess
    
    # Vitis AI compiler command for ZCU104
    compile_cmd = [
        "vai_c_xir",
        "--xmodel", xmodel_path,
        "--arch", "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json",
        "--output_dir", output_dir,
        "--net_name", "yolov11_zcu104"
    ]
    
    try:
        print(f"Running: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Compilation successful!")
            print(f"✓ Compiled model saved in: {output_dir}")
            return True
        else:
            print(f"❌ Compilation failed:")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Quantization for Vitis AI')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='Path to the model weights file')
    parser.add_argument('--calib_dir', type=str, required=True,
                       help='Path to calibration images directory')
    parser.add_argument('--quant_mode', type=str, choices=['calib', 'test'], default='calib',
                       help='Quantization mode: calib for calibration, test for deployment')
    parser.add_argument('--deploy', action='store_true',
                       help='Export deployment models (use with test mode)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile for ZCU104 FPGA (requires xmodel)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use: cpu or cuda')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Check if calibration directory exists
    if not os.path.exists(args.calib_dir):
        print(f"❌ Calibration directory not found: {args.calib_dir}")
        return
    
    # Run quantization
    success = quantize_model(
        model_path=args.model_path,
        calibration_data_dir=args.calib_dir,
        quant_mode=args.quant_mode,
        deploy=args.deploy,
        device=args.device
    )
    
    if not success:
        print("❌ Quantization failed!")
        return
    
    # Compile for FPGA if requested
    if args.compile:
        xmodel_path = "quantize_result/YOLOv11_int.xmodel"  # Default XMODEL path
        if os.path.exists(xmodel_path):
            compile_for_zcu104(xmodel_path)
        else:
            print(f"❌ XMODEL not found: {xmodel_path}")
            print("   Make sure to run with --deploy flag first")

if __name__ == "__main__":
    main()
