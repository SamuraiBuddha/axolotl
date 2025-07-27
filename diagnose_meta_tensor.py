"""
Diagnose the meta tensor issue
"""
import torch
import transformers
import peft
from packaging import version

print("Version Information:")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

print("\n" + "="*60)
print("Testing different loading methods:")
print("="*60)

model_name = "Qwen/Qwen2.5-Coder-7B"

# Method 1: Standard loading
print("\n1. Standard loading with device_map='auto':")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("✓ Success with device_map='auto'")
    # Check if model is on meta device
    first_param = next(model.parameters())
    print(f"First parameter device: {first_param.device}")
    print(f"First parameter dtype: {first_param.dtype}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}...")

# Method 2: Manual device placement
print("\n2. Loading to CPU then moving to CUDA:")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": "cpu"}  # Force CPU first
    )
    model = model.cuda()
    print("✓ Success with CPU->CUDA")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}...")

# Method 3: Direct CUDA loading
print("\n3. Direct CUDA loading:")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0}  # Direct to GPU 0
    )
    print("✓ Success with direct GPU loading")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}...")

# Method 4: Check for offload folder
print("\n4. Checking for offload issues:")
import os
offload_folder = os.path.expanduser("~/.cache/huggingface/accelerate")
if os.path.exists(offload_folder):
    print(f"Offload folder exists: {offload_folder}")
    print("Contents:", os.listdir(offload_folder) if os.listdir(offload_folder) else "Empty")
else:
    print("No offload folder found")

# Method 5: Test with smaller model
print("\n5. Testing with smaller model (Qwen2.5-Coder-1.5B):")
try:
    small_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B",
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✓ Smaller model loads successfully")
    print(f"Device: {next(small_model.parameters()).device}")
    del small_model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}...")

print("\n" + "="*60)
print("Recommendations based on results above")
print("="*60)
