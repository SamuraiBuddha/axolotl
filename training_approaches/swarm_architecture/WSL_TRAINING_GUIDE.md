# WSL Training Guide for Swarm Architecture

## Quick Start Commands

Open your WSL Ubuntu terminal and run these commands:

### 1. Set up Python Environment

```bash
# Check if conda is installed
conda --version

# If not installed, install miniconda:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create swarm environment
conda create -n swarm python=3.11 -y
conda activate swarm
```

### 2. Install Dependencies

```bash
# PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers==4.52.2 accelerate==1.9.0 peft==0.7.1 bitsandbytes==0.41.3
pip install datasets sentencepiece protobuf scipy

# Install axolotl
cd ~
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

### 3. Copy Swarm Architecture to WSL

```bash
# Create workspace
mkdir -p ~/swarm_training
cd ~/swarm_training

# Copy from Windows (adjust path if needed)
cp -r /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture .

# Navigate to swarm directory
cd swarm_architecture
```

### 4. Download Base Models

```bash
# Create model download script
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os

models = [
    ("HuggingFaceTB/SmolLM-135M", "models/smollm-135m"),
    ("HuggingFaceTB/SmolLM-360M", "models/smollm-360m"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "models/qwen2.5-0.5b")
]

for model_id, local_dir in models:
    print(f"Downloading {model_id}...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(model_id, local_dir=local_dir)
    print(f"✓ Downloaded to {local_dir}")
EOF

python download_models.py
```

### 5. Configure Accelerate

```bash
# Run accelerate config
accelerate config

# Choose these options:
# - No distributed training
# - No DeepSpeed
# - No FullyShardedDataParallel
# - Do not use FP16/BF16 (we use 4-bit quantization)
# - GPU: 0
```

### 6. Update Configs for Local Models

```bash
# Update paths in configs to use downloaded models
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' intent_parser/config.yaml
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' context_manager/config.yaml
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' error_recognizer/config.yaml
sed -i 's|Qwen/Qwen2.5-0.5B-Instruct|./models/qwen2.5-0.5b|g' api_mappers/docker_mapper_config.yaml
sed -i 's|HuggingFaceTB/SmolLM-360M|./models/smollm-360m|g' orchestrator/config.yaml
```

### 7. Train Models

```bash
# Train intent parser first (smallest, good test)
echo "Training Intent Parser..."
accelerate launch -m axolotl.cli.train intent_parser/config.yaml

# If successful, train others
echo "Training Error Recognizer..."
accelerate launch -m axolotl.cli.train error_recognizer/config.yaml

echo "Training Context Manager..."
accelerate launch -m axolotl.cli.train context_manager/config.yaml

# Or use the automated script
python scripts/train_swarm.py
```

### 8. Monitor Training

```bash
# Watch GPU usage (if NVIDIA)
watch -n 1 nvidia-smi

# Check training progress
tail -f intent_parser/output/training.log
```

### 9. Test Trained Models

```bash
# Run test suite
python scripts/test_swarm.py

# Run interactive demo
python scripts/swarm_coordinator.py
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Memory Issues
```bash
# Reduce batch size in configs
sed -i 's/micro_batch_size: 4/micro_batch_size: 1/g' */config.yaml
```

### Path Issues
```bash
# Ensure we're using absolute paths
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Expected Training Times

With a decent GPU (RTX 3060 or better):
- Intent Parser: ~15-20 minutes
- Error Recognizer: ~15-20 minutes  
- Context Manager: ~10-15 minutes
- API Mappers: ~10 minutes each
- Orchestrator: ~30-40 minutes

Total: ~2 hours for full swarm

## Success Indicators

1. Loss decreasing steadily (starting ~2-3, ending ~0.5-1)
2. Model checkpoints saved in `output/` directories
3. No CUDA out of memory errors
4. Test accuracy > 85% for each component

## Next Steps

After training completes:

1. Copy trained models back to Windows:
```bash
cp -r */output /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/trained_swarm_models/
```

2. Update swarm coordinator to use trained models:
```python
config = {
    "models": {
        "intent_parser": {
            "path": "./trained_swarm_models/intent_parser/output",
            "enabled": True
        },
        # ... etc
    }
}
```

3. Deploy the swarm!

---

**Remember**: WSL2 gives you a full Linux environment with proper Python package compatibility. This is the recommended way to train models on Windows machines.
