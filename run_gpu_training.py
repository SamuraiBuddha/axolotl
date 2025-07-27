"""
Set environment and run training with GPU
"""
import os
import subprocess

print("Setting up environment for GPU training...")

# Set environment variables
env_vars = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_LAUNCH_BLOCKING": "1",  # For debugging
    "TRANSFORMERS_OFFLINE": "0",
    "HF_DATASETS_OFFLINE": "0",
}

for key, value in env_vars.items():
    os.environ[key] = value
    print(f"Set {key}={value}")

# Clear cache
print("\nClearing CUDA cache...")
import torch
torch.cuda.empty_cache()
print("✓ Cache cleared")

print("\nDevice check:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\nStarting training...")
print("Command: python -m axolotl.cli.train bcl_physics_gpu_final.yaml")

# Run training
try:
    subprocess.run([
        "python", "-m", "axolotl.cli.train", "bcl_physics_gpu_final.yaml"
    ], check=True)
except subprocess.CalledProcessError as e:
    print(f"\nTraining failed with error code {e.returncode}")
    print("Try running directly:")
    print("python -m axolotl.cli.train bcl_physics_gpu_final.yaml")
