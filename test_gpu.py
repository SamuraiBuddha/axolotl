"""
Quick GPU verification script
Run this after installing CUDA PyTorch
"""

import torch
import subprocess

print("=" * 60)
print("PyTorch GPU Verification")
print("=" * 60)

# Basic PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test tensor operations
    print("\nTesting GPU tensor operations...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ Matrix multiplication successful on GPU")
    
    # Memory usage
    print(f"\nCurrent GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
else:
    print("\n❌ CUDA is NOT available!")
    print("This means PyTorch cannot use your GPU.")
    
# Check bitsandbytes
print("\n" + "=" * 60)
print("Bitsandbytes Check")
print("=" * 60)

try:
    import bitsandbytes as bnb
    print(f"✓ Bitsandbytes version: {bnb.__version__}")
    
    # Test 4-bit quantization
    print("\nTesting 4-bit quantization...")
    linear = torch.nn.Linear(100, 100)
    linear_4bit = bnb.nn.Linear4bit(100, 100)
    print("✓ 4-bit layer creation successful")
    
except Exception as e:
    print(f"❌ Bitsandbytes error: {e}")

# nvidia-smi output
print("\n" + "=" * 60)
print("NVIDIA-SMI Output")
print("=" * 60)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
except:
    print("❌ Could not run nvidia-smi")

print("\n" + "=" * 60)
print("Ready for GPU training!" if torch.cuda.is_available() else "GPU setup needed!")
print("=" * 60)
