#!/bin/bash
# Swarm Architecture WSL2 Setup Script
# Run this in your WSL2 Ubuntu environment

echo "=== Setting up Swarm Architecture in WSL2 ==="
echo "This script will set up the complete training environment"
echo

# Check if we're in WSL2
if ! grep -q microsoft /proc/version; then
    echo "Warning: This doesn't appear to be WSL2"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up directories
WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
PROJECT_BASE="/mnt/c/Users/${WINDOWS_USER}/Documents/GitHub/axolotl"
SWARM_DIR="${PROJECT_BASE}/training_approaches/swarm_architecture"

echo "Project base: $PROJECT_BASE"
echo "Swarm directory: $SWARM_DIR"

# Check if directory exists
if [ ! -d "$SWARM_DIR" ]; then
    echo "Error: Swarm directory not found at $SWARM_DIR"
    echo "Please check the path and try again"
    exit 1
fi

cd "$SWARM_DIR"

# Install system dependencies
echo
echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git curl wget

# Create Python 3.11 virtual environment
echo
echo "=== Creating Python 3.11 virtual environment ==="
python3.11 -m venv swarm_env
source swarm_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust for your CUDA version)
echo
echo "=== Installing PyTorch ==="
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For CUDA 11.8 (uncomment if you have GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo
echo "=== Installing core dependencies ==="
pip install transformers==4.41.0
pip install accelerate==0.30.0
pip install peft==0.11.0
pip install bitsandbytes==0.43.0
pip install datasets==2.19.0
pip install sentencepiece
pip install wandb
pip install scipy
pip install einops

# Install axolotl
echo
echo "=== Installing Axolotl ==="
cd /tmp
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
cd "$SWARM_DIR"

# Download base models
echo
echo "=== Downloading base models ==="
mkdir -p models

# SmolLM-135M
if [ ! -d "models/smollm-135m" ]; then
    echo "Downloading SmolLM-135M..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir='models/smollm-135m')"
fi

# SmolLM-360M
if [ ! -d "models/smollm-360m" ]; then
    echo "Downloading SmolLM-360M..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM-360M', local_dir='models/smollm-360m')"
fi

# Qwen2.5-0.5B
if [ ! -d "models/qwen2.5-0.5b" ]; then
    echo "Downloading Qwen2.5-0.5B-Instruct..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b')"
fi

# Update configs to use local models
echo
echo "=== Updating config files ==="
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' intent_parser/config.yaml
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' context_manager/config.yaml
sed -i 's|HuggingFaceTB/SmolLM-135M|./models/smollm-135m|g' error_recognizer/config.yaml
sed -i 's|HuggingFaceTB/SmolLM-360M|./models/smollm-360m|g' orchestrator/config.yaml
sed -i 's|Qwen/Qwen2.5-0.5B-Instruct|./models/qwen2.5-0.5b|g' api_mappers/docker_mapper_config.yaml

# Create training script
cat > train_swarm_wsl.py << 'EOF'
#!/usr/bin/env python3
"""
Swarm Training Script for WSL2
Trains all micro-models in the swarm architecture
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def train_model(name, config_path):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Config: {config_path}")
    print(f"Started: {datetime.now()}")
    print(f"{'='*60}")
    
    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {name} trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} training failed: {e}")
        return False

def main():
    # Configure accelerate
    print("Configuring accelerate...")
    subprocess.run(["accelerate", "config", "default"], check=True)
    
    # Models to train
    models = [
        ("Intent Parser", "intent_parser/config.yaml"),
        ("Error Recognizer", "error_recognizer/config.yaml"),
        ("Context Manager", "context_manager/config.yaml"),
        ("Docker API Mapper", "api_mappers/docker_mapper_config.yaml"),
        ("Orchestrator", "orchestrator/config.yaml"),
    ]
    
    results = {}
    for name, config in models:
        if Path(config).exists():
            results[name] = train_model(name, config)
        else:
            print(f"Config not found: {config}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    successful = sum(results.values())
    total = len(results)
    print(f"Models trained: {successful}/{total}")
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {model}")

if __name__ == "__main__":
    main()
EOF

chmod +x train_swarm_wsl.py

# Create test script
cat > test_swarm_wsl.py << 'EOF'
#!/usr/bin/env python3
"""
Test trained swarm models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def test_model(model_path, test_input):
    """Test a trained model"""
    print(f"\nTesting model at: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
        
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test generation
    inputs = tokenizer(test_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Output: {response}")

def main():
    print("=== Testing Swarm Models ===")
    
    # Test intent parser
    test_model(
        "intent_parser/output",
        "User: create a new python file\nAssistant:"
    )
    
    # Test error recognizer
    test_model(
        "error_recognizer/output",
        "Error: FileNotFoundError: No such file or directory\nAnalysis:"
    )

if __name__ == "__main__":
    main()
EOF

chmod +x test_swarm_wsl.py

echo
echo "=== Setup Complete! ==="
echo
echo "Environment activated. To train the swarm:"
echo "  python train_swarm_wsl.py"
echo
echo "To test trained models:"
echo "  python test_swarm_wsl.py"
echo
echo "To activate environment in future sessions:"
echo "  cd $SWARM_DIR"
echo "  source swarm_env/bin/activate"
echo
echo "Current environment: $(which python)"
echo "Python version: $(python --version)"
