"""
Fix meta tensor issues
"""
import os
import shutil
import torch

print("Attempting to fix meta tensor issues...")

# 1. Clear PyTorch cache
print("\n1. Clearing PyTorch cache...")
torch.cuda.empty_cache()
print("✓ PyTorch cache cleared")

# 2. Clear Hugging Face cache (optional - ask user first)
print("\n2. Hugging Face cache locations:")
hf_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache):
    size = sum(os.path.getsize(os.path.join(dirpath, filename))
              for dirpath, dirnames, filenames in os.walk(hf_cache)
              for filename in filenames) / (1024**3)
    print(f"Cache size: {size:.2f} GB at {hf_cache}")
    
    # Check for accelerate offload
    offload_path = os.path.join(hf_cache, "accelerate")
    if os.path.exists(offload_path):
        print(f"Found accelerate offload folder: {offload_path}")
        response = input("Delete accelerate offload folder? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(offload_path)
            print("✓ Offload folder deleted")

# 3. Set environment variables
print("\n3. Setting environment variables...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("✓ Environment variables set")

# 4. Create a minimal working config
print("\n4. Creating minimal BCL config...")
minimal_config = """base_model: Qwen/Qwen2.5-Coder-7B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: ./bcl_complete_training.jsonl
    type: alpaca

output_dir: ./outputs/bcl_minimal_test
val_set_size: 0.05

# Minimal settings
sequence_len: 512
micro_batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 1
learning_rate: 0.0002

# Simple adapter
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj

# Basic training settings
warmup_steps: 10
logging_steps: 10
eval_steps: 50
save_steps: 100

# Force CPU if needed
# device: cpu  # Uncomment to force CPU

# No special features
gradient_checkpointing: false
sample_packing: false
wandb_mode: disabled
chat_template: qwen3
"""

with open("bcl_minimal_config.yaml", "w") as f:
    f.write(minimal_config)
print("✓ Created bcl_minimal_config.yaml")

print("\n5. Testing model load with minimal config...")
print("Run: python diagnose_meta_tensor.py")
print("Then try: python -m axolotl.cli.train bcl_minimal_config.yaml")
