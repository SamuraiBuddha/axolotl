#!/bin/bash
# Quick setup commands for WSL Ubuntu

echo "Setting up Swarm Training in WSL Ubuntu..."

# Navigate to home
cd ~

# Create training directory
mkdir -p swarm_training
cd swarm_training

# Copy from Windows
echo "Copying swarm architecture from Windows..."
cp -r /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture .

# Check if copied successfully
if [ -d "swarm_architecture" ]; then
    echo "✓ Files copied successfully"
    echo "✓ Found $(find swarm_architecture -name "*.jsonl" | wc -l) training data files"
    echo "✓ Found $(find swarm_architecture -name "config.yaml" | wc -l) model configs"
else
    echo "✗ Copy failed - check path"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Install conda if needed:"
echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
echo "   bash Miniconda3-latest-Linux-x86_64.sh"
echo ""
echo "2. Create environment:"
echo "   conda create -n swarm python=3.11 -y"
echo "   conda activate swarm"
echo ""
echo "3. Install dependencies:"
echo "   pip install torch transformers accelerate peft bitsandbytes"
echo "   git clone https://github.com/OpenAccess-AI-Collective/axolotl ~/axolotl"
echo "   cd ~/axolotl && pip install -e . && cd -"
echo ""
echo "4. Start training:"
echo "   cd ~/swarm_training/swarm_architecture"
echo "   python scripts/train_swarm.py"
