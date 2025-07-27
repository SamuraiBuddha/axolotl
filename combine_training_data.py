"""
Combine original BCL training data with physics training battery
Creates a comprehensive dataset for both syntax and physics understanding
"""

import json

print("Combining BCL training datasets...")

# Read original training data
original_data = []
with open('bcl_training_data_fixed.jsonl', 'r') as f:
    for line in f:
        original_data.append(json.loads(line))

# Read physics training data
physics_data = []
with open('bcl_physics_training_battery.jsonl', 'r') as f:
    for line in f:
        physics_data.append(json.loads(line))

# Combine datasets
combined_data = original_data + physics_data

# Shuffle to mix syntax and physics examples
import random
random.shuffle(combined_data)

# Save combined dataset
with open('bcl_complete_training.jsonl', 'w') as f:
    for item in combined_data:
        f.write(json.dumps(item) + '\n')

print(f"Original BCL examples: {len(original_data)}")
print(f"Physics examples: {len(physics_data)}")
print(f"Total combined examples: {len(combined_data)}")

# Show distribution of instruction types
instruction_types = {}
for item in combined_data:
    inst = item['instruction']
    instruction_types[inst] = instruction_types.get(inst, 0) + 1

print("\nInstruction type distribution:")
for inst, count in sorted(instruction_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {count:4d} - {inst}")

# Create a new config for extended training
config_content = """base_model: Qwen/Qwen2.5-Coder-7B

strict: false
chat_template: qwen3

datasets:
  - path: ./bcl_complete_training.jsonl
    type: alpaca

val_set_size: 0.05
output_dir: ./outputs/bcl_physics_enhanced
dataset_prepared_path: last_run_prepared

sequence_len: 2048
sample_packing: false
eval_sample_packing: false
pad_to_sequence_len: false

# GPU configuration (with A4000)
load_in_4bit: true
adapter: qlora
lora_r: 64  # Increased for more complex physics understanding
lora_alpha: 128
lora_dropout: 0.05  # Add dropout for better generalization
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - down_proj
  - up_proj
  - gate_proj  # Added for better reasoning

wandb_project: bcl-physics-enhanced
wandb_name: qwen-bcl-physics-complete

gradient_accumulation_steps: 8  # Increased for larger effective batch
micro_batch_size: 2
num_epochs: 2  # Less epochs needed with more diverse data
optimizer: adamw_torch_4bit
lr_scheduler: cosine
learning_rate: 0.0001  # Slightly lower for fine-tuning on top of existing

bf16: true
tf32: true

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

logging_steps: 20
flash_attention: false  # Still no Windows Triton support

warmup_ratio: 0.1  # 10% warmup
evals_per_epoch: 4
saves_per_epoch: 2
weight_decay: 0.01

# Early stopping to prevent overfitting
early_stopping_patience: 3
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
"""

with open('bcl_physics_config.yaml', 'w') as f:
    f.write(config_content)

print("\nCreated bcl_physics_config.yaml for enhanced training")
print("\nTo train with physics understanding:")
print("python -m axolotl.cli.train bcl_physics_config.yaml")
