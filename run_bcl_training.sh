#!/bin/bash
# Setup script for BCL fine-tuning

# 1. Copy your training data
echo "Copy bcl_training_data.jsonl to the axolotl directory"

# 2. Install dependencies if needed
pip install -e . 

# 3. Install flash-attention (optional but recommended)
pip install flash-attn --no-build-isolation

# 4. Run training
accelerate launch -m axolotl.cli.train bcl_finetune_config.yaml