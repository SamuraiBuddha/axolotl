#!/bin/bash
# Axolotl installation script for WSL2
# This handles all the dependency issues

echo "=== Installing Axolotl in WSL2 ==="
echo "This will set up axolotl with all dependencies properly"
echo

# Check virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "ERROR: No virtual environment active!"
    echo "Run: source swarm_env/bin/activate"
    exit 1
fi

echo "Using virtual environment: $VIRTUAL_ENV"
echo

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential python3-dev git

# Upgrade pip and install build tools
echo "Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch first (required by many dependencies)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies in order
echo "Installing core dependencies..."
pip install packaging
pip install numpy
pip install transformers
pip install accelerate
pip install peft
pip install datasets

# Install bitsandbytes without CUDA if on CPU
echo "Installing bitsandbytes..."
pip install bitsandbytes

# Clone axolotl
echo "Cloning axolotl..."
cd /tmp
rm -rf axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl

# Install axolotl without dependencies first
cd axolotl
echo "Installing axolotl..."
pip install --no-deps -e .

# Install remaining requirements selectively
echo "Installing remaining requirements..."
# Skip packages that cause issues
for package in $(cat requirements.txt | grep -v flash-attn | grep -v deepspeed | grep -v apex); do
    pip install "$package" || echo "Skipped: $package"
done

# Go back to project
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture

echo
echo "=== Installation Complete ==="
echo "Test with: python -c 'import axolotl; print(\"Axolotl imported successfully\")'"
echo

# Test import
python -c 'import axolotl; print("✓ Axolotl imported successfully!")'

if [ $? -eq 0 ]; then
    echo
    echo "Ready to train with axolotl!"
    echo "Run: python train_swarm_wsl.py"
else
    echo
    echo "Axolotl import failed. Try the simple training approach instead:"
    echo "Run: python simple_train_wsl.py"
fi
